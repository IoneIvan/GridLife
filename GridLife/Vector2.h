#pragma once
#include <cmath>
#include <iostream>

class Vector2 {
public:
    float x;
    float y;

    // Constructors
    Vector2() : x(0), y(0) {}
    Vector2(float x, float y) : x(x), y(y) {}

    // Static properties
    static Vector2 zero() { return Vector2(0, 0); }
    static Vector2 one() { return Vector2(1, 1); }
    static Vector2 up() { return Vector2(0, 1); }
    static Vector2 down() { return Vector2(0, -1); }
    static Vector2 left() { return Vector2(-1, 0); }
    static Vector2 right() { return Vector2(1, 0); }

    // Methods
    float magnitude() const {
        return std::sqrt(x * x + y * y);
    }
    void setVector(float newX, float newY)
    {
        x = newX;
        y = newY;
    }
    Vector2 normalized() const {
        float mag = magnitude();
        return (mag > 0) ? Vector2(x / mag, y / mag) : Vector2(0, 0);
    }

    // Vector addition
    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }

    // Vector subtraction
    Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }

    // Scalar multiplication
    Vector2 operator*(float scalar) const {
        return Vector2(x * scalar, y * scalar);
    }

    // Dot product
    float dot(const Vector2& other) const {
        return x * other.x + y * other.y;
    }

    // Distance between two vectors
    static float distance(const Vector2& a, const Vector2& b) {
        return (a - b).magnitude();
    }

    // Linear interpolation
    static Vector2 lerp(const Vector2& a, const Vector2& b, float t) {
        t = (t < 0) ? 0 : (t > 1) ? 1 : t; // Clamp t to [0, 1]
        return a + (b - a) * t;
    }

    // Print function for debugging
    void print() const {
        std::cout << "Vector2(" << x << ", " << y << ")\n";
    }
};
