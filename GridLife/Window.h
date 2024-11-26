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
#include "UIComponent.h"

class Window {
    UIComponent windowUI;
    UIComponent plane;
    UIComponent sideBar;
    UIComponent gameCanvas;
    UIComponent button;

    const float SIDEBAR_WIDTH = 100.0f; // Sidebar width in screen units
    const float BUTTON_HEIGHT = 20.0f;  // Button height in screen units
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
    void initializeUI()
    {
        // Initialize text renderer
        HDC hdc = GetDC(glfwGetWin32Window(window)); // Get device context from GLFW

        windowUI.uiRect = UIRect(Vector2(0, 0), Vector2(width, height), Vector2(), Vector2());
        windowUI.textRenderer.initFont(hdc, 28);

        gameCanvas.uiRect = UIRect(Vector2(0, 0), Vector2(0, 0), Vector2(0.2, 0), Vector2(1.0, 1.0), &windowUI.uiRect);
        gameCanvas.uiRenderer.color = Color(0.0, 0.0, 0.0);
        gameCanvas.textRenderer.initFont(hdc, 28);

        sideBar.uiRect = UIRect(Vector2(0, 0), Vector2(0, 0), Vector2(0.0, 0.0), Vector2(0.2, 1.0), &windowUI.uiRect);
        sideBar.uiRenderer.color = Color(0.2, 0.2, 0.2);
        sideBar.textRenderer.initFont(hdc, 28);

        button.uiRect = UIRect(Vector2(10, -5), Vector2(30, -20), Vector2(0.0, 1.0), Vector2(0.0, 1.0), &sideBar.uiRect);
        button.uiRenderer.color = Color(0.9, 0.9, 0.2);
        button.textRenderer.initFont(hdc, 10);

        ReleaseDC(glfwGetWin32Window(window), hdc); // Release the device context
    }
public:
    GLFWwindow* window;
    int width, height;

    Vector2 worldDimentionMin;
    Vector2 worldDimentionMax;
    Vector2 worldSize;

    float cellWidth, cellHeight;
    void setGeneration(int newGeneration)
    {
        generation = newGeneration;
    }


    Window(int w, int h, int Ww, int Wh, const char* title) {

        if (!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
            exit(EXIT_FAILURE);
        }
        worldSize.x = w;
        worldSize.y = h;
        worldDimentionMax.x = w / 2;
        worldDimentionMax.y = h/2;
        window = glfwCreateWindow(Ww, Wh, title, NULL, NULL);
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

        
        initializeUI();

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
        sideBar.uiRenderer.render();
        button.uiRenderer.render("Generation: " + std::to_string(generation));
        glFlush();
    }

    void renderGrid(Cell* grid) {
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);

        for (int y = worldDimentionMin.y; y < worldDimentionMax.y && y < worldSize.y; y++) {
            for (int x = worldDimentionMin.x; x < worldDimentionMax.x && x < worldSize.x; x++) {

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
                Vector2 scale = worldDimentionMax - worldDimentionMin;
                Vector2 cellMin(x * gameCanvas.uiRect.getDimentions().x / scale.x + gameCanvas.uiRect.getGlobalPosMin().x,
                    y * gameCanvas.uiRect.getDimentions().y / scale.y + gameCanvas.uiRect.getGlobalPosMin().y);
                Vector2 cellMax((x + 1) * gameCanvas.uiRect.getDimentions().x / scale.x + gameCanvas.uiRect.getGlobalPosMin().x,
                    (y + 1) * gameCanvas.uiRect.getDimentions().y / scale.y + gameCanvas.uiRect.getGlobalPosMin().y);

                glVertex2f(cellMin.x, cellMin.y); // Bottom left
                glVertex2f(cellMax.x, cellMin.y); // Bottom right
                glVertex2f(cellMax.x, cellMax.y); // Top right
                glVertex2f(cellMin.x, cellMax.y); // Top left
            }
        }


        glEnd();
        glFlush();
    }


};
