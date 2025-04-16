#include "mnist.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

// Read MNIST dataset
float* loadMNISTImages(const char* filename, int numImages) {
    if (VERBOSE) printf("Loading MNIST images from %s...\n", filename);
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float* images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i*INPUT_SIZE + j] = pixel / 255.0f;
        }
        if (VERBOSE && i % 10000 == 0) printf("Loaded %d images\n", i);
    }
    fclose(file);
    
    if (VERBOSE) {
        printf("First pixel of first image: %.4f\n", images[0]);
        printf("Last pixel of first image: %.4f\n", images[INPUT_SIZE-1]);
        printf("Image loading complete\n");
    }
    return images;
}

float* loadMNISTLabels(const char* filename, int numLabels) {
    if (VERBOSE) printf("Loading MNIST labels from %s...\n", filename);
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float* labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i*OUTPUT_SIZE + j] = (j == label) ? 1.0 : 0.0;
        }
        
        if (VERBOSE && i % 10000 == 0) printf("Loaded %d labels\n", i);
    }
    fclose(file);
    
    if (VERBOSE) {
        printf("First label: ");
        for (int j = 0; j < OUTPUT_SIZE; j++) printf("%.1f ", labels[j]);
        printf("\n");
        printf("Label loading complete\n");
    }
    return labels;
}