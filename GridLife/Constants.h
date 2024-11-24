#pragma once

#define NUM_GENES 16 // Define the number of genes each cell will have
#define GENES 32
#define REP_ENERGY 1000

extern int WIDTH;
extern int HEIGHT;
int WIDTH = 250;
int HEIGHT = 250; 

struct Cell {
    int mutation;
    int alive;        // Alive or dead state
    int age;          // Age of the cell
    int energy;
    int activeGene;
    int genes[NUM_GENES]; // Array of integers representing genes
};