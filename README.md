# 3230_multi

Optimize the matrix-vector-multiplication algorithm of GPT by multi-threading. Similar to other neural networks, GPT and its variations utilize matrix-vector-multiplication, or called fullyconnected/linear layer in DL, to apply the parameter learned, which takes >70% of the whole calculation. Thus, to accelerate the GPT and get faster response, it’s critical to have faster matrixvector-multiplication, and multi-threading are usually considered powerful.

In this assignment, I will use an open-source variation of GPT, llama2 released by Meta. I use pthread.h with mutex_lock + conditional variable to implement a multi-threading version of matrix-vector-multiplication. This
multi-threading version will significantly accelerate the inference of Large Language Model.
