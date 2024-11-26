#pragma once
#include "Vector2.h"
#include "UIRect.h"
#include "UIRenderer.h"
#include "TextRenderer.h"
#include <string>

class UIComponent
{
public:
	UIRect uiRect;
	UIRenderer uiRenderer;
	TextRenderer textRenderer;

	UIComponent() { uiRenderer.rect = &uiRect; uiRenderer.textRenderer = &textRenderer;
	};

};