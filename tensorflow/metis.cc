#include <assert.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/util/tensor_format.h"
#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif
#include <metis.h>
#ifdef __cplusplus
}
#endif

using std::vector;
using namespace tensorflow;
using shape_inference::Shape;
using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

REGISTER_OP("Metis")
    .Input("adj: bool")
    .Input("weights: float32")
    .Input("n_clusters: int64")
    .Output("supernode_assign: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {        
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class MetisOp : public OpKernel {
 public:
      explicit GraclusOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& adj = context->input(0);
    auto adj_arr = adj.tensor<bool, 3>();
    const Tensor& weights = context->input(1);
    auto weights_arr = weights.tensor<float, 3>();
    const Tensor& n_clusters = context->input(2);
    auto n_clusters_arr = n_clusters.tensor<int, 1>();
    // Create an output tensor
    Tensor* supernode_assign = NULL;
      
    const TensorShape& adj_shape = adj.shape();
    TensorShape out_shape = adj_shape;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &supernode_assign));
    
    auto output_shape = out_shape.dim_sizes();
    auto output_flat = supernode_assign->flat<int>();

    // Set all but the first element of the output tensor to 0.
    const int N = output_flat.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    auto supernode_assign_arr = supernode_assign->tensor<int, 3>();

    const int batch_size = output_shape[0];
    std::function<void(int64, int64)> shard;
    shard = [&adj_arr, &weights_arr, &n_clusters_arr, &supernode_assign_arr, &output_shape](int64 start, int64 limit) {
        for (int graph = start; graph < limit; ++graph) {
            int n_clus = n_clusters_arr(graph);
            std::vector<int> xadj;
            xadj.push_back(0);
            std::vector<int> adjncy;
            std::vector<int> adjwgt;
            for (int m = 0; m < output_shape[1]; m++) {
                for (int n = 0; n < output_shape[0]; n++) {
                    if (m != n && adj_arr(graph, m, n)) {
                        adjncy.push_back(n);
                        adjwgt.push_back(weights_arr(graph, m, n));
                    }
                }
                xadj.push_back(xadj.size());
            }
            idx_t options[METIS_NOPTIONS];
            graph_t *graph;
            idx_t *part;
            idx_t objval;
            params_t *params;
            int status=0;
            part = imalloc(graph->nvtxs, "main: part");
            METIS_SetDefaultOptions(options);
            
            idxtype* assign;
            
            for(int i = 0; i < adjncy.size(); i++) {
                supernode_assign_arr(graph, i, assign[i]) = 1.;
            }
            free(assign);
        }
    };

    // This is just a very crude approximation
    const int64 single_cost = 10000 * output_shape[1] * output_shape[2];

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size, single_cost, shard);
  }

 private:
    TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("Metis").Device(DEVICE_CPU), MetisOp);
