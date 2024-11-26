#include "Input.h"

// Static member initialization
GLFWwindow* Input::windowPtr = nullptr;
std::unordered_map<int, bool> Input::keyStates;
std::unordered_map<int, bool> Input::previousKeyStates;
std::unordered_map<int, bool> Input::mouseButtonStates;
std::unordered_map<int, bool> Input::previousMouseButtonStates;
std::pair<double, double> Input::mousePosition = { 0.0, 0.0 };