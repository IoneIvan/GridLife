#pragma once
class Color
{
public:
	float r;
	float g;
	float b;

	Color(float r = 0.0, float g = 0.0, float b = 0.0) :r(r), g(g), b(b) {}
	Color(Color& c) :r(c.r), g(c.g), b(c.b) {}
};