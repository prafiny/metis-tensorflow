#include <assert.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/util/tensor_format.h"
#include <iostream>

extern "C"
{
#include <metis.h>
}

using std::vector;
using namespace tensorflow;
using shape_inference::Shape;
using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

REGISTER_OP("Metis")
    .Input("adj: bool")
    .Input("weights: int32")
    .Input("n_clusters: int32")
    .Output("supernode_assign: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {        
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class MetisOp : public OpKernel {
 public:
      explicit MetisOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& adj = context->input(0);
    auto adj_arr = adj.tensor<bool, 3>();
    const Tensor& weights = context->input(1);
    auto weights_arr = weights.tensor<int, 3>();
    const Tensor& n_clusters = context->input(2);
    auto n_clusters_arr = n_clusters.tensor<int, 1>();
    // Create an output tensor
    Tensor* supernode_assign = NULL;
      
    const TensorShape& adj_shape = adj.shape();
    TensorShape out_shape = adj_shape;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &supernode_assign));
    
    auto output_shape = out_shape.dim_sizes();
    auto output_flat = supernode_assign->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = output_flat.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    auto supernode_assign_arr = supernode_assign->tensor<float, 3>();

    const int batch_size = output_shape[0];
    std::function<void(int64, int64)> shard;
    shard = [&adj_arr, &weights_arr, &n_clusters_arr, &supernode_assign_arr, &output_shape](int64 start, int64 limit) {
        int i;
        for (int graph = start; graph < limit; ++graph) {
            int n_clus = 100;//n_clusters_arr(graph);
            std::vector<idx_t> xadj;
            xadj.push_back(0);
            std::vector<idx_t> adjncy;
            std::vector<idx_t> adjwgt;
            std::vector<int> ids;
            for (int m = 0; m < output_shape[1]; m++) {
                if (adj_arr(graph, m, m)) {
                    ids.push_back(m);
                }
            }
            if (ids.size() > 0) {
                if (n_clus > 1) {
                    for (int m = 0; m < ids.size(); m++) {
                        for (int n = 0; n < ids.size(); n++) {
                            if (m != n && adj_arr(graph, ids[m], ids[n])) {
                                adjncy.push_back(n);
                                adjwgt.push_back(weights_arr(graph, ids[m], ids[n])+1);
                            }
                        }
                        xadj.push_back(adjncy.size());
                    }
                    idx_t options[METIS_NOPTIONS];
                    idx_t objval;
                    int status=0;
                    idx_t edgecut;
                    METIS_SetDefaultOptions(options);
                    options[METIS_OPTION_NUMBERING] = 0; // C-style numbering

                    int nvtxs = xadj.size()-1;
                    int ncon = 1;

                    idx_t *part = new idx_t[nvtxs];
                    status = METIS_PartGraphKway(&nvtxs, &ncon, &xadj.front(), 
                            &adjncy.front(), NULL, NULL, &adjwgt.front(), 
                            &n_clus, NULL, NULL, options, 
                            &edgecut, part);

                    for(int i = 0; i < ids.size(); i++) {
                        supernode_assign_arr(graph, ids[i], ids[part[i]]) = 1.;
                    }
                    delete part;
                }
                else {
                    for(i = 0; i < ids.size(); i++) {
                        supernode_assign_arr(graph, ids[i], ids[i]) = 1.;
                    }
                }
            }
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
