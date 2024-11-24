#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept> // For std::runtime_error
#include <cmath>
#include <cstdlib> // For rand() and srand()
#include <iostream>
#include <random>
#include <functional>  // For std::hash

#include "Constants.h"
class Window {

    float getDeterministicRandomFloat(int value) {
        // Create a hash from the input value to use as the seed
        std::hash<int> hashFunction;
        size_t seed = hashFunction(value);

        // Use the seed with a random number generator
        std::mt19937 generator(seed);  // Mersenne Twister engine
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        // Return a random float between 0 and 1
        return distribution(generator);
    }
public:
    GLFWwindow* window;
    int width, height;
    float cellWidth, cellHeight;

    Window(int w, int h, const char* title) {
        if (!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
            exit(EXIT_FAILURE);
        }

        window = glfwCreateWindow(w, h, title, NULL, NULL);
        if (!window) {
            fprintf(stderr, "Failed to open GLFW window\n");
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);
        glewInit();

        glfwGetWindowSize(window, &width, &height);
        cellWidth = (float)width / WIDTH;
        cellHeight = (float)height / HEIGHT;

        // Set up the viewport and projection
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, WIDTH, 0.0, HEIGHT, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
    }

    ~Window() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    bool shouldClose() {
        return glfwWindowShouldClose(window);
    }

    void swapBuffers() {
        glfwSwapBuffers(window);
    }

    void pollEvents() {
        glfwPollEvents();
    }

    void renderGrid(Cell* grid) {
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int idx = y * WIDTH + x;
                float r = grid[idx].energy > 0 ? grid[idx].energy / float(REP_ENERGY) : 0.0f; // Alive cells are white, dead cells are black
                float g = 0;
                float b = 0;
                
              
                if (r > 0)
                {
                    //g = grid[idx].mutation / 3000.0;

                    b = grid[idx].genes[0] / float(GENES);
                    g = grid[idx].mutation / 1000.0f;
               }
                
                glColor3f(r, g, b);
                glVertex2f((float)x, (float)y); // Bottom left
                glVertex2f((float)(x + 1), (float)y); // Bottom right
                glVertex2f((float)(x + 1), (float)(y + 1)); // Top right
                glVertex2f((float)x, (float)(y + 1)); // Top left
            }
        }

        glEnd();
        glFlush();
    }

};
