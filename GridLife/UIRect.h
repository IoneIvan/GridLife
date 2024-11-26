#pragma once
#include "Vector2.h"

class UIRect {
public:
	UIRect* parent;
	Vector2 posMin;
	Vector2 posMax;
	Vector2 anchMin;
	Vector2 anchMax;

	UIRect()
	{

	}
	UIRect(Vector2 posMin, Vector2 posMax, Vector2 anchMin, Vector2 anchMax, UIRect* parent = nullptr)
		: posMin(posMin), posMax(posMax), anchMin(anchMin), anchMax(anchMax), parent(parent)
	{

	}
	void operator=(UIRect& second)
	{
		parent = second.parent;
		posMin = second.posMin;
		posMax = second.posMax;
		anchMin = second.anchMin;
		anchMax = second.anchMax;
	}
	Vector2 getDimentions()
	{
		Vector2 out;
		if (parent != nullptr)
		{
			out.x = anchMax.x * parent->getDimentions().x - anchMin.x * parent->getDimentions().x;
			out.y = anchMax.y * parent->getDimentions().y - anchMin.y * parent->getDimentions().y;
		}
		out.x += posMax.x - posMin.x;
		out.y += posMax.y - posMin.y;
		
		return out;
	}
	Vector2 getGlobalPosMin()
	{
		Vector2 out;
		if (parent != nullptr)
		{
			out.x = anchMin.x * parent->getDimentions().x;
			out.y = anchMin.y * parent->getDimentions().y;
		}
		out.x += posMin.x;
		out.y += posMin.y;

		return out;
	}
	Vector2 getGlobalPosMax()
	{
		Vector2 out;
		if (parent != nullptr)
		{
			out.x = anchMax.x * parent->getDimentions().x;
			out.y = anchMax.y * parent->getDimentions().y;
		}
		out.x += posMax.x;
		out.y += posMax.y;

		return out;
	}
};