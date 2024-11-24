#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept> // For std::runtime_error
#include <cmath>
#include <cstdlib> // For rand() and srand()
#include <iostream>
#include <random>
#include <functional>  // For std::hash

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "Constants.h"
#include "TextRenderer.h"
class Window {
    const float SIDEBAR_WIDTH = 100.0f; // Sidebar width in screen units
    const float BUTTON_HEIGHT = 20.0f;  // Button height in screen units
    TextRenderer textRenderer;
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
    int generation = 0;
    bool isShowSideBar = false;
public:
    GLFWwindow* window;
    int width, height;
    float cellWidth, cellHeight;
    void setGeneration(int newGeneration)
    {
        generation = newGeneration;
    }
    void showSideBar()
    {
        isShowSideBar = true;
    }
    void hideSideBar()
    {
        isShowSideBar = false;
    }
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

        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, WIDTH, 0.0, HEIGHT, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);

        // Initialize text renderer
        HDC hdc = GetDC(glfwGetWin32Window(window)); // Get device context from GLFW
        textRenderer.initFont(hdc);
        ReleaseDC(glfwGetWin32Window(window), hdc); // Release the device context
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


    void renderSidebar() {
        // Render sidebar background
        glColor3f(0.2f, 0.2f, 0.2f); // Dark gray color
        glBegin(GL_QUADS);
        glVertex2f(0.0f, 0.0f);
        //glVertex2f(SIDEBAR_WIDTH, 0.0f);
        //glVertex2f(SIDEBAR_WIDTH, HEIGHT);
        glVertex2f(WIDTH, 0.0f);
        glVertex2f(WIDTH, HEIGHT);
        glVertex2f(0.0f, HEIGHT);
        glEnd();

        // Render button
        glColor3f(0.6f, 0.6f, 0.6f); // Light gray button
        glBegin(GL_QUADS);
        glVertex2f(10.0f, HEIGHT - 30.0f); // Top left
        glVertex2f(SIDEBAR_WIDTH - 10.0f, HEIGHT - 30.0f); // Top right
        glVertex2f(SIDEBAR_WIDTH - 10.0f, HEIGHT - 30.0f - BUTTON_HEIGHT); // Bottom right
        glVertex2f(10.0f, HEIGHT - 30.0f - BUTTON_HEIGHT); // Bottom left
        glEnd();

        // Render labels
        textRenderer.renderText(10.0f, HEIGHT - 60.0f, "Generation: " + std::to_string(generation));
        textRenderer.renderText(10.0f, HEIGHT - 80.0f, "Label 2");
        textRenderer.renderText(10.0f, HEIGHT - 100.0f, "Label 3");
        glFlush();

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
