cd build
ninja
cd ..

rm -rf pass_logs_gemm
python demo_presentation.py --kernel gemm --dump


python analyze_sync.py --kernel gemm


python visualize_dep_graph.py body_graph.log pass_logs_gemm/dep_graph_gemm

python visualize_pipeline.py
mv body_pipeline_schedule.png pass_logs_gemm/pipeline_gemm.png

python bufferspan_new.py --kernel gemm
mv buffer_new.png pass_logs_gemm/buffer_gemm.png


