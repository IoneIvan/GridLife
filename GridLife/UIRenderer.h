#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "UIRect.h"
#include "TextRenderer.h"
#include "Color.h"
#include <string>

class UIRenderer
{
public:
	UIRect* rect;
	TextRenderer* textRenderer;
	Color color;
	bool isRender = true;
	UIRenderer()
	{
		color = Color(0, 0, 0);
	}
	
	void render(std::string label = "")
	{
		glColor3f(color.r, color.g, color.b);

		glBegin(GL_QUADS);
		Vector2 min = rect->getGlobalPosMin();
		Vector2 max = rect->getGlobalPosMax();
		glVertex2f(min.x, min.y);
		glVertex2f(max.x, min.y);
		glVertex2f(max.x, max.y);
		glVertex2f(min.x, max.y);
		glEnd();

		textRenderer->renderText(min.x, min.y, label);

		glEnd();
	}
};