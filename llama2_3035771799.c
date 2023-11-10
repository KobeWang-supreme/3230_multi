/*
PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME:
* NAME:
* UID :
* Development Platform:
* Remark: (How much you implemented?)
* How to compile: (gcc -o llama2_[UID] llama2_[UID].c utilities.c -O2 -pthread -lm)

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o llama2_[UID] llama2_[UID].c utilities.c -O2 -pthread -lm

Then Run with:
$ ./llama2_[UID] <seed> <thr_count>
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-Vector Multiplication, used in Attention and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each row, so we can use Multi-Threading for acceleration.
 * 
 * Please use <pthread.h> and your favorite synchronization method,
 * semaphore / mutex lock + conditional variable
 * 
 * A sequential version is provided in seq.c, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// Addtional Header File Here
#include <pthread.h>

// Global Variables
struct rusage main_usage;        // get usage for main thread
// create a list of threads
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_var2 = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_var3 = PTHREAD_COND_INITIALIZER;
int active_threads = 0;
int num_threads = 0;
struct rusage *thr_usages; // get usage for each thread
int finish = 0;
pthread_t *threads;

typedef struct {
    int id;
    float* out;
    float* vec;
    float* mat;
    int col;
    int row;
    int status;
    struct timeval user_t;
    struct timeval sys_t;
} thread_data;

thread_data* data;

void* thr_func(void* arg);

int init_mat_vec_mul(int thr_count) {
    // initialize mutex and conditional variables
    num_threads = thr_count;
    threads = malloc(thr_count * sizeof(pthread_t));
    thr_usages = (struct rusage*) malloc(thr_count * sizeof(struct rusage));
    data = (thread_data*) malloc(thr_count * sizeof(thread_data));
    
    // create threads
    for (int i = 0; i < thr_count; ++i) {
        data[i].id = i;
        pthread_create(&threads[i], NULL, thr_func, &data[i]);
    }

}

void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    // assign new parameters to threads
    
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < num_threads; ++i) {
        data[i].out = out;
        data[i].vec = vec;
        data[i].mat = mat;
        data[i].col = col;
        data[i].row = row;
        data[i].status = 1;
        data[i].user_t.tv_sec = 1;
        data[i].user_t.tv_usec = 1;
        data[i].sys_t.tv_sec = 1;
        data[i].sys_t.tv_usec = 1;
        ++active_threads;
        
        pthread_cond_signal(&cond_var);
    }
    // wait for threads to finish
    while (active_threads > 0) {
        
        pthread_cond_wait(&cond_var2, &mutex);
        
    }
    pthread_mutex_unlock(&mutex);
    return;
}

int close_mat_vec_mul() {
    pthread_mutex_lock(&mutex);
    
    // Wake up threads to collect the system usage (of themselves)
    for (int i = 0; i < num_threads; ++i) {
        data[i].status = 2;
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_cond_signal(&cond_var);
    }

    // wait for threads to finish
    while (finish < num_threads) {
        pthread_cond_wait(&cond_var3, &mutex);
    }
    pthread_mutex_unlock(&mutex);

    // Print system usage of each thread.
    for (int i = 0; i < num_threads; ++i) {
        printf("Thread %d has completed - user: %ld.%04ld s, system: %ld.%04ld s\n", i, data[i].user_t.tv_sec, data[i].user_t.tv_usec / 100, data[i].sys_t.tv_sec, data[i].sys_t.tv_usec / 100);
        fflush(stdout); 
    }
    // print system usage of main thread
    getrusage(RUSAGE_THREAD, &main_usage);
    printf("Main thread - user: %ld.%04ld s, system: %ld.%04ld s\n", main_usage.ru_utime.tv_sec, main_usage.ru_utime.tv_usec / 100, main_usage.ru_stime.tv_sec, main_usage.ru_stime.tv_usec / 100);
    fflush(stdout);
    // Clear resources.
    free(data);
    free(threads);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_var);
    pthread_cond_destroy(&cond_var2);
    pthread_cond_destroy(&cond_var3);
    free(thr_usages);
    return 0;
}

void *thr_func(void *arg) {
    // get thread id
    thread_data* dat = (thread_data*) arg;
    int ratio;
    int start;
    int end;
    while(1){
        // wait for main thread to assign new parameters
        pthread_mutex_lock(&mutex);
        while (dat->status == 0) {
            pthread_cond_wait(&cond_var, &mutex);
        }
        pthread_mutex_unlock(&mutex);
        // check if thread should close
        if (dat->status == 2) {
            //printf("Thread %d closing\n", dat->id);
            //fflush(stdout);
            // printf("111111\n");
            fflush(stdout);
            pthread_mutex_lock(&mutex);
            finish++;
            
            getrusage(RUSAGE_THREAD, &thr_usages[dat->id]);
            data[dat->id].user_t = thr_usages[dat->id].ru_utime;
            data[dat->id].sys_t = thr_usages[dat->id].ru_stime;
            pthread_cond_signal(&cond_var3);
            pthread_mutex_unlock(&mutex);
            return NULL;
        }
        
        int k = dat->id;
        
        // Perform matrix-vector multiplication here.
        
        if (dat->row % num_threads != 0) {

            ratio = dat->row / num_threads + 1;

            if (k == num_threads - 1) {
                start = k * ratio;
                end = dat->row;
            }
            else{
                start = k * ratio;
                end = (k + 1) * ratio;
            }
        }
        else{

            ratio = dat->row / num_threads;
            start = k * ratio;
            end = (k + 1) * ratio;   
        }

        for (int i = start; i < end; i++) {
            float val = 0.0f;
            for (int j = 0; j < dat->col; j++) {
                val += dat->mat[i * dat->col + j] * dat->vec[j]; 
            }
            dat->out[i] = val;
        }
        
        // inform main thread that this thread has finished
        pthread_mutex_lock(&mutex);
        dat->status = 0;
        --active_threads;
        
        pthread_cond_signal(&cond_var2);
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
       
}

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();
    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}
