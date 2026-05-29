#!/bin/bash
cd build
ninja
cd ..

rm -rf pass_logs_flashattn
python demo_presentation.py --kernel flashattn --dump

python analyze_sync.py --kernel flashattn


python visualize_dep_graph.py body_graph.log pass_logs_flashattn/dep_graph_flashattn

python visualize_pipeline.py
mv body_pipeline_schedule.png pass_logs_flashattn/pipeline_flashattn.png

python bufferspan_new.py --kernel flashattn
mv buffer_new.png pass_logs_flashattn/buffer_flashattn.png



