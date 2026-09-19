/**
    GPU single-linkage hierarchical tree wrapper. Compile with:

    nvcc -O3 -std=c++17 --shared \
    -x cu \
    -Xcompiler -fPIC \
    -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
    $(python -m pybind11 --includes) \
    gpu_linkage.cpp \
    -o gpu_linkage$(python3-config --extension-suffix) \
    -I/opt/conda/include \
    -I/opt/conda/include/rapids \
    -L/opt/conda/lib \
    -lcuvs -lrmm
*/

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

#include <cuvs/cluster/agglomerative.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace py = pybind11;

namespace {

struct MST {
    py::object rows;
    py::object cols;
    py::object values;

    MST(py::object rows_, py::object cols_, py::object values_)
        : rows(std::move(rows_)),
          cols(std::move(cols_)),
          values(std::move(values_))
    {
    }
};


// Get the raw CUDA device pointer from a CuPy ndarray.
std::uintptr_t cupy_ptr(const py::object& a)
{
    return a.attr("data").attr("ptr").cast<std::uintptr_t>();
}


// Return the current CuPy CUDA stream pointer.
//
// CuPy exposes this through:
//     cupy.cuda.get_current_stream().ptr
//
// Using the current CuPy stream means that operations queued by CuPy before
// this call and operations queued by CuPy after this call have the expected
// stream ordering.
cudaStream_t cupy_current_stream()
{
    py::module_ cupy = py::module_::import("cupy");

    py::object stream =
        cupy.attr("cuda").attr("get_current_stream")();

    std::uintptr_t ptr = stream.attr("ptr").cast<std::uintptr_t>();

    return reinterpret_cast<cudaStream_t>(ptr);
}


py::object cupy_empty_1d(
    const py::module_& cupy,
    std::int64_t n,
    const char* dtype)
{
    return cupy.attr("empty")(
        py::make_tuple(n),
        py::arg("dtype") = dtype
    );
}


py::object cupy_empty_2d(
    const py::module_& cupy,
    std::int64_t n_rows,
    std::int64_t n_cols,
    const char* dtype)
{
    return cupy.attr("empty")(
        py::make_tuple(n_rows, n_cols),
        py::arg("dtype") = dtype
    );
}


cuvs::distance::DistanceType parse_metric(const std::string& metric)
{
    if (metric == "L2Expanded")
        return cuvs::distance::DistanceType::L2Expanded;

    if (metric == "L2SqrtExpanded")
        return cuvs::distance::DistanceType::L2SqrtExpanded;

    if (metric == "CosineExpanded")
        return cuvs::distance::DistanceType::CosineExpanded;

    if (metric == "L1")
        return cuvs::distance::DistanceType::L1;

    throw std::invalid_argument(
        "Unsupported metric '" + metric +
        "'. Supported metrics are: "
        "L2Expanded, L2SqrtExpanded, CosineExpanded, L1."
    );
}


py::tuple build_linkage(
    py::object X,
    int c,
    const std::string& metric_name)
{
    py::module_ cupy = py::module_::import("cupy");

    // ---------------------------------------------------------------------
    // Validate that X is a CuPy ndarray.
    // ---------------------------------------------------------------------

    py::object cupy_ndarray = cupy.attr("ndarray");

    if (!py::isinstance(X, cupy_ndarray)) {
        throw std::invalid_argument(
            "X must be a CuPy ndarray."
        );
    }

    // ---------------------------------------------------------------------
    // Validate dtype, dimensionality, and layout.
    // ---------------------------------------------------------------------

    std::string dtype =
        X.attr("dtype").attr("name").cast<std::string>();

    if (dtype != "float32") {
        throw std::invalid_argument(
            "X must have dtype float32."
        );
    }

    int ndim = X.attr("ndim").cast<int>();

    if (ndim != 2) {
        throw std::invalid_argument(
            "X must be a two-dimensional CuPy array."
        );
    }

    bool c_contiguous =
        X.attr("flags").attr("c_contiguous").cast<bool>();

    if (!c_contiguous) {
        throw std::invalid_argument(
            "X must be C-contiguous (row-major). "
            "Use cupy.ascontiguousarray(X) first."
        );
    }

    py::tuple shape = X.attr("shape").cast<py::tuple>();

    const std::int64_t n_rows =
        shape[0].cast<std::int64_t>();
    
    const std::int64_t n_cols =
        shape[1].cast<std::int64_t>();

    if (c < 0) {
        throw std::invalid_argument(
            "c must be non-negative."
        );
    }

    // ---------------------------------------------------------------------
    // Allocate all five C++ output parameters as CuPy arrays.
    //
    // build_linkage() requires:
    //
    //   out_mst
    //   dendrogram
    //   out_distances
    //   out_sizes
    //   core_dists
    //
    // For ordinary distance-based KNN linkage, core_dists is std::nullopt.
    // ---------------------------------------------------------------------

    const std::int64_t nnz = n_rows - 1;

    // MST COO components.
    py::object mst_rows =
        cupy_empty_1d(cupy, nnz, "int64");

    py::object mst_cols =
        cupy_empty_1d(cupy, nnz, "int64");

    py::object mst_values =
        cupy_empty_1d(cupy, nnz, "float32");

    // Dendrogram: [n_rows - 1, 2].
    py::object dendrogram =
        cupy_empty_2d(cupy, nnz, 2, "int64");

    // Merge distances/heights.
    py::object distances =
        cupy_empty_1d(cupy, nnz, "float32");

    // Cluster sizes.
    py::object sizes =
        cupy_empty_1d(cupy, nnz, "int64");

    // ---------------------------------------------------------------------
    // Obtain raw device pointers.
    // ---------------------------------------------------------------------

    auto* X_ptr =
        reinterpret_cast<const float*>(cupy_ptr(X));

    auto* mst_rows_ptr =
        reinterpret_cast<std::int64_t*>(cupy_ptr(mst_rows));

    auto* mst_cols_ptr =
        reinterpret_cast<std::int64_t*>(cupy_ptr(mst_cols));

    auto* mst_values_ptr =
        reinterpret_cast<float*>(cupy_ptr(mst_values));

    auto* dendrogram_ptr =
        reinterpret_cast<std::int64_t*>(cupy_ptr(dendrogram));

    auto* distances_ptr =
        reinterpret_cast<float*>(cupy_ptr(distances));

    auto* sizes_ptr =
        reinterpret_cast<std::int64_t*>(cupy_ptr(sizes));

    // ---------------------------------------------------------------------
    // Use CuPy's current CUDA stream for RAFT/cuVS.
    // ---------------------------------------------------------------------

    cudaStream_t stream = cupy_current_stream();

    raft::resources res{};

    raft::resource::set_cuda_stream(
        res,
        rmm::cuda_stream_view(stream)
    );

    // ---------------------------------------------------------------------
    // Construct non-owning RAFT views over the CuPy arrays.
    // ---------------------------------------------------------------------

    auto X_view =
        raft::make_device_matrix_view<
            const float,
            std::int64_t,
            raft::row_major>(
                X_ptr,
                n_rows,
                n_cols
            );

    auto dendrogram_view =
        raft::make_device_matrix_view<
            std::int64_t,
            std::int64_t,
            raft::row_major>(
                dendrogram_ptr,
                nnz,
                2
            );

    auto distances_view =
        raft::make_device_vector_view<
            float,
            std::int64_t>(
                distances_ptr,
                nnz
            );

    auto sizes_view =
        raft::make_device_vector_view<
            std::int64_t,
            std::int64_t>(
                sizes_ptr,
                nnz
            );

    // The COO structure consists of the row and column index arrays.
    auto mst_structure =
        raft::make_device_coordinate_structure_view<
            std::int64_t,
            std::int64_t,
            std::size_t>(
                mst_rows_ptr,
                mst_cols_ptr,
                n_rows,
                n_rows,
                static_cast<std::size_t>(nnz)
            );

    auto mst_view =
        raft::make_device_coo_matrix_view<
            float,
            std::int64_t,
            std::int64_t,
            std::size_t>(
                mst_values_ptr,
                mst_structure
            );

    // ---------------------------------------------------------------------
    // Configure the KNN-graph linkage.
    // ---------------------------------------------------------------------

    namespace agglomerative =
        cuvs::cluster::agglomerative;

    namespace linkage_params =
        agglomerative::helpers::linkage_graph_params;

    linkage_params::distance_params params;

    params.dist_type =
        agglomerative::Linkage::KNN_GRAPH;

    params.c = c;

    const auto metric = parse_metric(metric_name);

    // ---------------------------------------------------------------------
    // Call the actual cuVS implementation.
    //
    // For ordinary distance-based single linkage, core_dists is absent.
    // ---------------------------------------------------------------------

    {
        py::gil_scoped_release release;

        agglomerative::helpers::build_linkage(
            res,
            X_view,
            params,
            metric,
            mst_view,
            dendrogram_view,
            distances_view,
            sizes_view,
            std::nullopt
        );

        // Make sure all work associated with this invocation has completed
        // before the RAFT resource object is destroyed.
        raft::resource::sync_stream(res);
    }

    // ---------------------------------------------------------------------
    // Package the COO MST as a small Python object.
    // ---------------------------------------------------------------------

    MST mst(
        std::move(mst_rows),
        std::move(mst_cols),
        std::move(mst_values)
    );

    // The Python return value corresponds one-for-one to the five C++
    // output parameters:
    //
    //   1. out_mst       -> MST object
    //   2. dendrogram    -> CuPy array
    //   3. out_distances -> CuPy array
    //   4. out_sizes     -> CuPy array
    //   5. core_dists    -> None
    //
    return py::make_tuple(
        py::cast(std::move(mst)),
        std::move(dendrogram),
        std::move(distances),
        std::move(sizes),
        py::none()
    );
}

} // anonymous namespace


PYBIND11_MODULE(gpu_linkage, m)
{
    m.doc() =
        "Minimal pybind11 wrapper around "
        "cuvs::cluster::agglomerative::helpers::build_linkage.";

    py::class_<MST>(m, "MST")
        .def_readonly("rows", &MST::rows)
        .def_readonly("cols", &MST::cols)
        .def_readonly("values", &MST::values);

    m.def(
        "build_linkage",
        &build_linkage,
        py::arg("X"),
        py::arg("c") = 15,
        py::arg("metric") = "L2Expanded",
        R"doc(
Build a single-linkage hierarchy using a KNN graph.

Parameters
----------
X : cupy.ndarray
    C-contiguous float32 matrix of shape (n_samples, n_features),
    resident on the current CUDA device.

c : int, default=15
    KNN-graph parameter. cuVS chooses k approximately as log(n) + c.

metric : str, default="L2Expanded"
    Distance metric.

Returns
-------
mst : MST
    COO representation of the MST. All three fields are CuPy arrays:
        mst.rows
        mst.cols
        mst.values

dendrogram : cupy.ndarray
    int64 array of shape (n_samples - 1, 2).

distances : cupy.ndarray
    float32 array of shape (n_samples - 1), containing the merge
    distances/heights.

sizes : cupy.ndarray
    int64 array of shape (n_samples - 1), containing cluster sizes.

core_dists : None
    No core distances are produced for ordinary distance-based
    single-linkage.
)doc"
    );
}
