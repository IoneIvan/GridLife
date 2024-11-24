#pragma once
#include <windows.h>  // For Windows API
#include <string>     // For std::string
#include <GL/glew.h>  // For OpenGL functions
#include <iostream>
class TextRenderer {
    GLuint base; // Display list base for the font

public:
    TextRenderer() : base(0) {}

    // Initialize the font with the current device context
    void initFont(HDC hdc) {
        base = glGenLists(96); // Reserve 96 display lists (ASCII range 32-127)
        if (base == 0) {
            std::cout << "Failed to create font display lists.";
        }

        // Create a font and map it to OpenGL display lists
        HFONT font = CreateFont(
            -24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, ANSI_CHARSET,
            OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, FF_DONTCARE | DEFAULT_PITCH,
            "Courier New" // Change font name as needed
        );

        if (!font) {
            std::cout << "Failed to create font display lists.";
        }

        SelectObject(hdc, font);
        wglUseFontBitmaps(hdc, 32, 96, base);
        DeleteObject(font); // Delete the font after it’s loaded into OpenGL
    }

    // Render text at the specified position
    void renderText(float x, float y, const std::string& text) {
        if (base == 0) {
            std::cerr << "Font not initialized. Call initFont() first." << std::endl;
            return;
        }

        glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glRasterPos2f(x, y); // Set text position
        glListBase(base - 32); // Set the display list base (ASCII 32-127)
        glCallLists(static_cast<GLsizei>(text.length()), GL_UNSIGNED_BYTE, text.c_str()); // Render text

        glPopMatrix();
        glPopAttrib();
    }

    ~TextRenderer() {
        if (base != 0) {
            glDeleteLists(base, 96); // Delete font display lists
        }
    }
};
