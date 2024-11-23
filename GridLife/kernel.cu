#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include "Window.h"
#include "Constants.h"
__device__ curandState* d_randStates;
// Kernel to initialize random states
__global__ void initRandStates(curandState* randStates, int width, int height, unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        curand_init(seed, idx, 0, &randStates[idx]);
    }
}

__global__ void initializeCellsKernel(Cell* cells, int width, int height, curandState* randStates) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        curandState localState = randStates[idx]; // Fetch pre-initialized state

        cells[idx].alive = curand(&localState) % 20 == 0 ? 1 : 0; // Randomly set alive or dead      
        cells[idx].energy = 0;// Randomly set energy
        cells[idx].mutation = curand(&localState) % 1000;// Randomly set energy
        if (cells[idx].alive)
        {
            cells[idx].energy = curand(&localState) % REP_ENERGY / 2;// Randomly set energy
        }
        cells[idx].age = 0; // Initial age
        cells[idx].activeGene = 0; // Initial age
        for (int g = 0; g < NUM_GENES; g++) {
            cells[idx].genes[g] = curand(&localState) % GENES; // Random genes
        }

        randStates[idx] = localState; // Save state back
    }
}

__global__ void updateKernel(Cell* current, Cell* next, int width, int height, curandState* randStates) {
    
    int neighborOffsets[4][2] = {
       {0, -1}, // Up
       {0, 1},  // Down
       {-1, 0}, // Left
       {1, 0}   // Right
    };
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int idx = y * width + x;
        curandState localState = randStates[idx]; // Fetch pre-initialized state

        int aliveParents = 0;
        int aliveInstantNeighbors = 0;
        int energy = current[idx].energy;
        int parentsEnergy = 0;

        int parents[4];

       
        if (energy > 0) {
            for (int i = 0; i < NUM_GENES; ++i)
                next[idx].genes[i] = current[idx].genes[i];
            next[idx].mutation = current[idx].mutation;
            for (int i = 0; i < 4; i++) {
                int neighborX = x + neighborOffsets[i][0] * 2;
                int neighborY = y + neighborOffsets[i][1] * 2;

                // Check if the neighbor is within bounds and alive
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    if (current[neighborY * width + neighborX].energy > 0) {
                        parents[aliveParents] = i;
                        if (current[neighborY * width + neighborX].energy > REP_ENERGY)
                            aliveParents++;
                        parentsEnergy += current[neighborY * width + neighborX].energy;
                    }
                }
            }

            int neighbors_energy = 0;
            int acumulateNeigbhorEnergy = 0;
            int mutants = 0;
            for (int i = 0; i < 4; i++) {
                int neighborX = x + neighborOffsets[i][0];
                int neighborY = y + neighborOffsets[i][1];

                // Check if the neighbor is within bounds and alive
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    if (current[neighborY * width + neighborX].energy > 0) {
                        aliveInstantNeighbors++;
                        neighbors_energy += current[neighborY * width + neighborX].energy;
                        if (current[neighborY * width + neighborX].genes[current[neighborY * width + neighborX].activeGene] == 2)
                            acumulateNeigbhorEnergy += current[neighborY * width + neighborX].energy / 4;
                        if (current[neighborY * width + neighborX].genes[current[neighborY * width + neighborX].activeGene] == 4)
                            acumulateNeigbhorEnergy -= current[neighborY * width + neighborX].energy;
                        if (current[neighborY * width + neighborX].mutation != current[idx].mutation)
                            mutants++;
                    }
                }
            }

            if (aliveInstantNeighbors > 0 && energy >= REP_ENERGY)
                energy = energy / aliveInstantNeighbors;

            energy += acumulateNeigbhorEnergy;

            next[idx].activeGene  = (current[idx].activeGene + NUM_GENES + 1) % NUM_GENES;

            
            if (current[idx].genes[current[idx].activeGene] == 0)
            {
                energy += current[idx].genes[next[idx].activeGene];
                next[idx].activeGene = (next[idx].activeGene + NUM_GENES + 1) % NUM_GENES;
            }
            if (current[idx].genes[current[idx].activeGene] == 1)
                energy -= 1;
            if (current[idx].genes[current[idx].activeGene] == 2)
                energy /= aliveInstantNeighbors - 1;
            if (current[idx].genes[current[idx].activeGene] == 3)
            {
                next[idx].activeGene = (next[idx].activeGene + aliveInstantNeighbors + NUM_GENES + 1) % NUM_GENES;
            } 
            if (current[idx].genes[current[idx].activeGene] == 4)
            {
                //energy -= neighbors_energy/aliveInstantNeighbors;
            }
            if (current[idx].genes[current[idx].activeGene] == 5)
            {
                next[idx].activeGene = (next[idx].activeGene + mutants + NUM_GENES + 1) % NUM_GENES;
            }
            if (curand(&localState) % 1000 == 0)
                energy = 0;
            energy -= 1;

            if (energy <= 0 || energy>=2* REP_ENERGY)
                energy = 0;

        }
        else {
            
            for (int i = 0; i < 4; i++) {
                int neighborX = x + neighborOffsets[i][0];
                int neighborY = y + neighborOffsets[i][1];

                // Check if the neighbor is within bounds and alive
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    if (current[neighborY * width + neighborX].energy > REP_ENERGY) {
                        parents[aliveParents] = i;
                            aliveParents++;
                        parentsEnergy += current[neighborY * width + neighborX].energy;
                    }
                }
            }
            if (aliveParents >= 1)
            {
                energy = parentsEnergy / aliveParents / 2 - 1;
                energy = energy > 0 ? energy : 0;

                next[idx].activeGene = 0;

                for (int i = 0; i < 4 - 1; ++i) {
                    for (int j = 0; j < 4 - i - 1; ++j) {
                        int neighborX = x + neighborOffsets[j][0];
                        int neighborY = y + neighborOffsets[j][1];
                        int neighborX2 = x + neighborOffsets[j][0];
                        int neighborY2 = y + neighborOffsets[j][1];
                        if (current[neighborY * width + neighborX].energy < current[neighborY2 * width + neighborX2].energy) { // Sort in descending order
                            // Swap arr[j] and arr[j + 1]
                            int temp = parents[j];
                            parents[j] = parents[j + 1];
                            parents[j + 1] = temp;
                        }
                    }
                }
                    int j = parents[curand(&localState) %  aliveParents];
                for (int i = 0; i < NUM_GENES; ++i)
                {

                    int randomIndex = curand(&localState) % 10;
                    if (randomIndex <= 3)
                        randomIndex = 0;
                    else if (randomIndex <= 6)
                        randomIndex = 1;
                    else if (randomIndex <= 8)
                        randomIndex = 2;
                    else
                        randomIndex = 3;

                    int neighborX = x + neighborOffsets[j][0];
                    int neighborY = y + neighborOffsets[j][1];

                    next[idx].genes[i] = current[neighborY * width + neighborX].genes[i];
                    next[idx].mutation = current[neighborY * width + neighborX].mutation;
                }
                if (curand(&localState) % 40 == 0)
                {
                    next[idx].mutation = curand(&localState) % 1000;
                    next[idx].genes[curand(&localState) % NUM_GENES] = curand(&localState) % GENES;
                }

            }
        }
        next[idx].energy = energy;

        randStates[idx] = localState; // Save state back
    }
}

void initializeCellular(Cell* current) {
    // Initialize the grid with a simple pattern and random genes
    srand(time(NULL));
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            current[idx].alive = 0; // Dead cell
            current[idx].age = 0;  // Initial age
            for (int g = 0; g < NUM_GENES; g++) {
                current[idx].genes[g] = rand() % 10; // Random genes
            }
        }
    }

    // A glider pattern
    current[1 * WIDTH + 0].alive = 1;
    current[2 * WIDTH + 1].alive = 1;
    current[0 * WIDTH + 2].alive = 1;
    current[1 * WIDTH + 2].alive = 1;
    current[2 * WIDTH + 2].alive = 1;
}

int main() {
    Window window(800, 800, "Game of Life");

    Cell* current = (Cell*)malloc(WIDTH * HEIGHT * sizeof(Cell));
    Cell* next = (Cell*)malloc(WIDTH * HEIGHT * sizeof(Cell));

    initializeCellular(current);

    Cell* dev_current = 0;
    Cell* dev_next = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for Cell arrays
    cudaStatus = cudaMalloc((void**)&dev_current, WIDTH * HEIGHT * sizeof(Cell));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&dev_next, WIDTH * HEIGHT * sizeof(Cell));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }


    // Copy initial grid to GPU
    cudaStatus = cudaMemcpy(dev_current, current, WIDTH * HEIGHT * sizeof(Cell), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }

   

    // Initialize cells on the GPU
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaMalloc(&d_randStates, WIDTH * HEIGHT * sizeof(curandState));
    initRandStates << <numBlocks, threadsPerBlock >> > (d_randStates, WIDTH, HEIGHT, time(NULL));
    cudaDeviceSynchronize();

    // Use a seed for random number generation
    initializeCellsKernel << <numBlocks, threadsPerBlock >> > (dev_current, WIDTH, HEIGHT, d_randStates);
    cudaDeviceSynchronize();

    // Mouse state variables
    double mouseX, mouseY;
    int mouseButtonState;

    int a = 0;
    // Main loop
    while (!window.shouldClose()) {

        // Check mouse events
        {
            glfwGetCursorPos(window.window, &mouseX, &mouseY); // Get mouse position
            mouseButtonState = glfwGetMouseButton(window.window, GLFW_MOUSE_BUTTON_LEFT); // Left mouse button state

            if (mouseButtonState == GLFW_PRESS) {
                // Convert mouse position to grid coordinates
                int gridX = (int)(mouseX / window.cellWidth);
                int gridY = HEIGHT - (int)(mouseY / window.cellHeight); // Flip Y axis since OpenGL has the origin at the bottom left
                int radius = 50;
                // Ensure the coordinates are within grid bounds
                if (gridX >= 0 && gridX < WIDTH && gridY >= 0 && gridY < HEIGHT) {
                    // Set cells within the radius to alive
                    for (int dx = -radius; dx <= radius; dx++) {
                        for (int dy = -radius; dy <= radius; dy++) {
                            // Calculate the distance from the center
                            if (dx * dx + dy * dy <= radius * radius) {
                                int newX = gridX + dx;
                                int newY = gridY + dy;

                                // Ensure the new coordinates are within grid bounds
                                if (newX >= 0 && newX < WIDTH && newY >= 0 && newY < HEIGHT) {
                                    // Set the cell to alive at the calculated position
                                    current[newY * WIDTH + newX].energy = REP_ENERGY / 2;
                                    for (int i = 0; i < NUM_GENES; ++i)
                                    {
                                        //TO DO make the genes random value from 0 to GENES
                                        current[newY * WIDTH + newX].genes[i] = rand() % (GENES); // Random value from 0 to GENES
                                    }
                                    printf("Cell alive at (%d, %d)\n", newX, newY);
                                }
                            }
                        }
                    }
                    cudaMemcpy(dev_current, current, WIDTH * HEIGHT * sizeof(Cell), cudaMemcpyHostToDevice);
                }
            }
        }
        // Check keyboard events
        {
            if (glfwGetKey(window.window, GLFW_KEY_A) == GLFW_PRESS) {
                a++; // Increase a by 1 when 'A' is pressed
                printf("Variable a increased to: %d\n", a);
            }
            if (glfwGetKey(window.window, GLFW_KEY_Q) == GLFW_PRESS) {
                a--; // Decrease a by 1 when 'Q' is pressed
                if (a < 0)
                    a = 0;
                printf("Variable a decreased to: %d\n", a);
            }
        }
        // Run the simulation for a number of generations
        for (int generation = 0; generation < a; generation++) {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

            unsigned long long seed = time(NULL)  + generation; // Change seed for each generation
            updateKernel << <numBlocks, threadsPerBlock >> > (dev_current, dev_next, WIDTH, HEIGHT, d_randStates);
           
            cudaDeviceSynchronize();

            // Swap the buffers
            Cell* temp = dev_current;
            dev_current = dev_next;
            dev_next = temp;
        }

        // Copy the current generation back to the host
            cudaMemcpy(current, dev_current, WIDTH * HEIGHT * sizeof(Cell), cudaMemcpyDeviceToHost);


        window.renderGrid(current);
        window.swapBuffers();
        window.pollEvents();
    }

    // Clean up
    cudaFree(dev_current);
    cudaFree(dev_next);
    free(current);
    free(next);

    return 0;
}
