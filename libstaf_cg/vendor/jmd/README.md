# vendor/jmd — CG origami CUDA for libstaf_cg

Copied from `neuralmdGPU/DEV/CG_and_WCA_LJ2_inter/src/` and compiled in
`libstaf_cg/CMakeLists.txt` (no prebuilt `.o`).

`nn_nn_mlp.cu` is the TF→ORT patch of `nn_nn.cu`: dual cutoff
(`Rc_inter` / `Rs_inter` / `Ra_inter`) and `Map_intra` / `Color_type_map` /
`map_color_interaction` are kept. WCA/LJ are omitted.
