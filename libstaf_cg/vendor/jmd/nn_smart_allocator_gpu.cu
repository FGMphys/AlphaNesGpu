#include <stdlib.h>
#include <cuda_runtime.h>
#include "nn_smart_allocator.h"
#include <cuda.h>
// Funzione per creare una matrice irregolare 2D
__host__ int** createIrregularMatrix2D_CUDA_int(int dim1, int* dim2, int factor) {
    // Calcola la dimensione totale necessaria
    int totaldim = 0;
    for (int k = 0; k < dim1; k++) {
        totaldim += dim2[k];
    }
    totaldim *= factor;

    // Allocazione di memoria per i puntatori host
    int** h_array = (int**)malloc(dim1 * sizeof(int*));

    // Allocazione di memoria per i dati totali sulla GPU
    int* d_data;
    cudaMalloc((void**)&d_data, totaldim * sizeof(int));

    // Allocazione di memoria per i puntatori sulla GPU
    int** d_array;
    cudaMalloc((void**)&d_array, dim1 * sizeof(int*));

    // Calcolo degli offset per ogni riga e copia dei puntatori su GPU
    h_array[0] = d_data;
    for (int j = 1; j < dim1; j++) {
        h_array[j] = h_array[j - 1] + dim2[j - 1] * factor;
    }

    // Copia dei puntatori sull'array irregolare sulla GPU
    cudaMemcpy(d_array, h_array, dim1 * sizeof(int*), cudaMemcpyHostToDevice);

    // Libera la memoria host
    //free(h_array);

    return h_array; // Ritorna il puntatore sulla GPU
}

// Funzione per liberare la memoria allocata sulla GPU
__host__ void freeIrregularMatrix2D_CUDA_int(int** d_array, int* d_data) {
    cudaFree(d_array);
    cudaFree(d_data);
}
// Funzione per creare una matrice irregolare 2D
__host__ double** createIrregularMatrix2D_CUDA(int dim1, int* dim2, int factor) {
    // Calcola la dimensione totale necessaria
    int totaldim = 0;
    for (int k = 0; k < dim1; k++) {
        totaldim += dim2[k];
    }
    totaldim *= factor;

    // Allocazione di memoria per i puntatori host
    double** h_array = (double**)malloc(dim1 * sizeof(double*));

    // Allocazione di memoria per i dati totali sulla GPU
    double* d_data;
    cudaMalloc((void**)&d_data, totaldim * sizeof(double));

    // Allocazione di memoria per i puntatori sulla GPU
    double** d_array;
    cudaMalloc((void**)&d_array, dim1 * sizeof(double*));

    // Calcolo degli offset per ogni riga e copia dei puntatori su GPU
    h_array[0] = d_data;
    for (int j = 1; j < dim1; j++) {
        h_array[j] = h_array[j - 1] + dim2[j - 1] * factor;
    }

    // Copia dei puntatori sull'array irregolare sulla GPU
    cudaMemcpy(d_array, h_array, dim1 * sizeof(double*), cudaMemcpyHostToDevice);

    // Libera la memoria host
//free(h_array);

    return h_array; // Ritorna il puntatore sulla GPU
}

// Funzione per liberare la memoria allocata sulla GPU
__host__ void freeIrregularMatrix2D_CUDA(double** d_array, double* d_data) {
    cudaFree(d_array);
    cudaFree(d_data);
}
