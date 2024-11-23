#pragma once
#define WIDTH 250
#define HEIGHT 250

#define NUM_GENES 4 // Define the number of genes each cell will have
#define GENES 10
#define REP_ENERGY 100
struct Cell {
    int mutation;
    int alive;        // Alive or dead state
    int age;          // Age of the cell
    int energy;
    int activeGene;
    int genes[NUM_GENES]; // Array of integers representing genes
};