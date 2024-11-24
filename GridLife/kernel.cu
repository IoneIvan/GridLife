#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>

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

        cells[idx].energy = 0;// Randomly set energy
        cells[idx].activeGene = 0; // Initial age
        cells[idx].mutation = curand(&localState) % 1000;// Randomly set energy
        int isAlive = curand(&localState) % 20 == 0 ? 1 : 0; // Randomly set alive or dead      
        
        if (isAlive)
            cells[idx].energy = curand(&localState) % REP_ENERGY / 2;// Randomly set energy
        
        for (int g = 0; g < NUM_GENES; g++) 
            cells[idx].genes[g] = curand(&localState) % GENES; // Random genes
        

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

        //Copy Energy
        int energy = current[idx].energy;

        //Alive Cell
        if (energy > 0) {
            
            //Calculate Neibhors
            int neighborEnergy = 0;
            int neighborAlive = 0;
            int mutants = 0;
            int sharedEnergy = 0;
            int damageEnergy = 0;

            for (int i = 0; i < 4; i++) {
                int neighborX = x + neighborOffsets[i][0];
                int neighborY = y + neighborOffsets[i][1];

                // Check if the neighbor is within bounds and alive
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    if (current[neighborY * width + neighborX].energy > 0) {

                        neighborAlive++;
                        neighborEnergy += current[neighborY * width + neighborX].energy;
                        
                        //Neighbor Shared Energy
                        if (current[neighborY * width + neighborX].genes[current[neighborY * width + neighborX].activeGene] == 2)
                            sharedEnergy += current[neighborY * width + neighborX].energy / 4;
                        //Neighbor Attacked
                        if (current[neighborY * width + neighborX].genes[current[neighborY * width + neighborX].activeGene] == 4)
                            damageEnergy -= current[neighborY * width + neighborX].energy;
                        //Mutant Detected
                        if (current[neighborY * width + neighborX].mutation != current[idx].mutation)
                            mutants++;
                    }
                }
            }
            //Remove Energy for reproduction
            if (neighborAlive > 0 && energy >= REP_ENERGY)
                energy = energy / neighborAlive;

            //Processed shared and damged energy
            //energy += sharedEnergy;
            energy -= damageEnergy;

            int newActiveGene = 0;

            //Update Gene
            next[idx].activeGene = (current[idx].activeGene + NUM_GENES + 1) % NUM_GENES;
            //Process Cells functions
            switch (current[idx].genes[current[idx].activeGene])
            {
            case 0:
                energy += current[idx].genes[next[idx].activeGene];
                next[idx].activeGene = current[idx].genes[newActiveGene];

                break;
            case 1:
                energy -= 1;
                break;
            case 2:
                if(neighborAlive > 0)
                //energy /= neighborAlive;
                //energy -= 1;
                break;
            case 3:
                newActiveGene = (next[idx].activeGene + NUM_GENES + neighborAlive) % NUM_GENES;
                next[idx].activeGene = current[idx].genes[newActiveGene];
                break;
            case 4:
                //energy += neighborEnergy / neighborAlive / 2;
                break;
            case 5:
                newActiveGene = (next[idx].activeGene + NUM_GENES + mutants) % NUM_GENES;
                next[idx].activeGene = current[idx].genes[newActiveGene];
                break;
            case 6:
                newActiveGene = (next[idx].activeGene + NUM_GENES + 1) % NUM_GENES;
                next[idx].activeGene = current[idx].genes[newActiveGene];
                break;
            }

            //random chance to die
            if (curand(&localState) % 1000 == 0)
                energy = 0;

            // remove energy for stayin alive
            energy -= 1;

            //Check if the energy is in the proper range
            if (energy <= 0 || energy>=2* REP_ENERGY)
                energy = 0;

        }
        //Empty Cell
        else {
            int parents[4];
            int parentAlive = 0;
            int parentEnergy = 0;

            //Calculate Parents
            for (int i = 0; i < 4; i++) {
                int neighborX = x + neighborOffsets[i][0];
                int neighborY = y + neighborOffsets[i][1];

                // Check if the neighbor is within bounds and alive
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    if (current[neighborY * width + neighborX].energy > REP_ENERGY) {
                        parents[parentAlive] = i;
                        parentAlive++;
                        parentEnergy += current[neighborY * width + neighborX].energy;
                    }
                }
            }
            
            //Process Reproduction
            if (parentAlive > 0)
            {

                //Copy energy of parents
                energy = parentEnergy / parentAlive / 2 - 1;
                energy = energy > 0 ? energy : 0;
                //Copy Genes
                int j = parents[curand(&localState) % parentAlive];
                for (int i = 0; i < NUM_GENES; ++i)
                {
                    int neighborX = x + neighborOffsets[j][0];
                    int neighborY = y + neighborOffsets[j][1];

                    next[idx].genes[i] = current[neighborY * width + neighborX].genes[i];
                    current[idx].genes[i] = current[neighborY * width + neighborX].genes[i];
                    next[idx].mutation = current[neighborY * width + neighborX].mutation;
                    current[idx].mutation = current[neighborY * width + neighborX].mutation;
                }
                //Mutation
                if (curand(&localState) % 40 == 0)
                {
                    next[idx].mutation = curand(&localState) % 1000;
                    current[idx].mutation = next[idx].mutation;
                    int gn = curand(&localState) % NUM_GENES;
                    int gv = curand(&localState) % GENES;
                    next[idx].genes[gn] = gv;
                    current[idx].genes[gn] = gv;
                }

                //Set ActiveGene to 0
                next[idx].activeGene = 0;
            }
        }
        next[idx].energy = energy;

        randStates[idx] = localState; // Save state back
    }
}


int main() {
    if (false)
    {
        // Prompt the user to enter the width
        std::cout << "Enter the width: ";
        std::cin >> WIDTH;

        // Prompt the user to enter the height
        std::cout << "Enter the height: ";
        std::cin >> HEIGHT;

        // Optionally, you can print the values to confirm
        std::cout << "Width set to: " << WIDTH << std::endl;
        std::cout << "Height set to: " << HEIGHT << std::endl;
    }
    else
    {
        WIDTH = 200;
        HEIGHT = 200;
    }
    int cellSize = 3;
    Window window(WIDTH * cellSize, HEIGHT * cellSize, "Game of Life");

    Cell* current = (Cell*)malloc(WIDTH * HEIGHT * sizeof(Cell));

    srand(time(NULL));

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
    cudaMemcpy(dev_next, dev_current, WIDTH * HEIGHT * sizeof(Cell), cudaMemcpyDeviceToDevice);

    // Mouse state variables
    double mouseX, mouseY;
    int mouseButtonState;

    int a = 200;
    bool isUpdateFrame = true;
    auto lastTime = std::chrono::high_resolution_clock::now(); // Start time
    int frameCount = 0; // Frame counter

    int generationCounter=0;
    // Main loop
    while (!window.shouldClose()) {

        // Check mouse events
        {
            glfwGetCursorPos(window.window, &mouseX, &mouseY); // Get mouse position
            mouseButtonState = glfwGetMouseButton(window.window, GLFW_MOUSE_BUTTON_LEFT); // Left mouse button state

            if (mouseButtonState == GLFW_PRESS && isUpdateFrame) {
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
            if (glfwGetKey(window.window, GLFW_KEY_O) == GLFW_PRESS) {
                a = 1; // Increase a by 1 when 'A' is pressed
                printf("Variable set to: %d\n", a);
            }
            if (glfwGetKey(window.window, GLFW_KEY_I) == GLFW_PRESS) {
                std::cout << "Enter generation per frame ( currently" << a << ") :";
                std::cin >> a;
                if (a <0)
                {
                    a = -a;
                    isUpdateFrame = !isUpdateFrame;
                    window.setGeneration(generationCounter);
                    window.showSideBar();
                }
                else
                {
                    window.hideSideBar();
                }
                printf("Variable set to: %d\n", a);
            }
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

            if (glfwGetKey(window.window, GLFW_KEY_R) == GLFW_PRESS ) {
                initializeCellsKernel << <numBlocks, threadsPerBlock >> > (dev_current, WIDTH, HEIGHT, d_randStates);
                cudaDeviceSynchronize();
                printf("Reset\n");
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

        generationCounter += a;

        if (isUpdateFrame)
        {
            // Copy the current generation back to the host
            cudaMemcpy(current, dev_current, WIDTH * HEIGHT * sizeof(Cell), cudaMemcpyDeviceToHost);
            window.renderGrid(current);
            window.swapBuffers();

        }
        else
        {
            window.setGeneration(generationCounter);
            window.renderSidebar();
            window.swapBuffers();

        }
            window.pollEvents();

        // Frame rate calculation
        frameCount+= a;
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - lastTime;

        // Calculate time per frame
        std::chrono::duration<double> frameTime = currentTime - lastTime; // Time taken for the current frame

        if (elapsed.count() >= 1.0) { // If one second has passed
            std::cout << "FPS: " << frameCount / elapsed.count() << std::endl; // Display the frame rate
            std::cout << "Time per frame: " << frameTime.count() << " seconds" << std::endl; // Display time per frame
            frameCount = 0; // Reset the frame count
            lastTime = currentTime; // Update the last time
        }
    }

    // Clean up
    cudaFree(dev_current);
    cudaFree(dev_next);
    free(current);

    return 0;
}
