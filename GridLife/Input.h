#pragma once
#include <GLFW/glfw3.h>
#include <unordered_map>
#include <iostream>

class Input {
public:
    // Initialize the input system
    static void Init(GLFWwindow* window) {
        windowPtr = window;
    }

    // Update the input state
    static void Update() {
        // Store the current key states
        for (int i = 0; i < 512; ++i) {
            if (glfwGetKey(windowPtr, i) == GLFW_PRESS) {
                keyStates[i] = true;
            }
        }

        // Store the current mouse button states
        for (int i = 0; i < 8; ++i) { // GLFW supports up to 8 mouse buttons
            if (glfwGetMouseButton(windowPtr, i) == GLFW_PRESS) {
                mouseButtonStates[i] = true;
            }
        }

        // Get the current mouse position
        double xpos, ypos;
        glfwGetCursorPos(windowPtr, &xpos, &ypos);
        mousePosition = { xpos, ypos };
    }

    // Check if a key is currently pressed
    static bool GetKey(int key) {
        return keyStates[key];
    }

    // Check if a key was just pressed
    static bool GetKeyDown(int key) {
        if (keyStates[key] && !previousKeyStates[key]) {
            return true;
        }
        return false;
    }

    // Check if a key was just released
    static bool GetKeyUp(int key) {
        if (!keyStates[key] && previousKeyStates[key]) {
            return true;
        }
        return false;
    }

    // Check if a mouse button is currently pressed
    static bool GetMouseButton(int button) {
        return mouseButtonStates[button];
    }

    // Check if a mouse button was just pressed
    static bool GetMouseButtonDown(int button) {
        if (mouseButtonStates[button] && !previousMouseButtonStates[button]) {
            return true;
        }
        return false;
    }

    // Check if a mouse button was just released
    static bool GetMouseButtonUp(int button) {
        if (!mouseButtonStates[button] && previousMouseButtonStates[button]) {
            return true;
        }
        return false;
    }

    // Get the current mouse position
    static std::pair<double, double> GetMousePosition() {
        return mousePosition;
    }

    // Call this at the end of each frame to update previous states
    static void EndFrame() {
        for (int i = 0; i < 512; ++i) {
            keyStates[i] = glfwGetKey(windowPtr, i) == GLFW_PRESS;
        }
        previousKeyStates = keyStates;

        for (int i = 0; i < 8; ++i) {
            mouseButtonStates[i] = glfwGetMouseButton(windowPtr, i) == GLFW_PRESS;
        }
        previousMouseButtonStates = mouseButtonStates;
    }

private:
    static GLFWwindow* windowPtr; // Pointer to the GLFW window
    static std::unordered_map<int, bool> keyStates;
    static std::unordered_map<int, bool> previousKeyStates;
    static std::unordered_map<int, bool> mouseButtonStates;
    static std::unordered_map<int, bool> previousMouseButtonStates;
    static std::pair<double, double> mousePosition; // Current mouse position
};

