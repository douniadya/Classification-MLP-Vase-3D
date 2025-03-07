
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np




def f(x):
    return 1.5 * np.exp(-0.42*x)*np.sin(x) + 0.05 * x + 0.8

N = 5000
vertices = []

for i in range(N):
    z = np.random.uniform(0, 6)
    phi = np.random.uniform(0, 2 * np.pi)
    rho = abs(f(z))
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    vertices.append((x, y, z))



def draw_point():

    glColor3f(0.8, 0.0, 5.8)  # Some kind of purple color
    glBegin(GL_POINTS)

    for v in vertices:
       glVertex3f(v[0], v[1], v[2])  # Point at the origin
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Set up perspective
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glPointSize(2)
    glEnable(GL_DEPTH_TEST)
    # Set gray background
    glClearColor(0.7, 0.7, 0.7, 1.0)

    running = True
    angle = 0.0
    radius = 10.0

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        eyeX = radius * np.cos(angle)
        eyeY = radius * np.sin(angle)
        eyeZ = 3.0

        # Use gluLookAt to position the camera
        gluLookAt(eyeX, eyeY, eyeZ,  # Camera position
                  0.0, 0.0, 3.0,  # Look-at point
                  0.0, 0.0, 1.0) 
        
        draw_point()
        pygame.display.flip()
        pygame.time.wait(10)
        angle += 0.01 
        if angle >= 2 * np.pi:
            angle = 0

    pygame.quit()

if __name__ == "__main__":
    main()
