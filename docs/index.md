# 184 Final Project Proposal: Watercolor Texture Rendering

## Summary & Team members
Team members: Andrew Wang, Evan Lohn, Johnathan Zhou  

Our final project idea is in the realm of non-photorealistic rendering. More specifically, we plan to utilize several layers of textures along with lighting to mimic & render the apperance of watercolor. 

## Problem Description

As more modern art move to digital forms, computer graphic rendering techniques to mimic traditional hand drawn art forms are becoming popular. One such art form is watercolors, where artists draw charming images through the blending of rich and vibrant colors. Our group was especially interested in the non-photorealistic rendering of watercolor-esque images. This marks a contrast from the typical photorealism focus in traditional computer graphics, allowing for more artistic flexibility and expression. However, this project has a big challenge in texture generation, where creating a new watercolor style texture for each mesh in the 3D image space could be difficult. To implement watercolor-esque non-photorealistic rendering, we will follow existing papers on the subject and utilize similar techinques. More specifically, we will (1) use existing project 1 code for rasterization and project 3 coding for lighting (2) introduce watercolor style texture generation and mapping to simulate brush wash like textures (3) create a pigment based lighting model to allow for shading with the set of colors like those found in typical watercolor paintings.

## Goals and Deliverables

(1) What we plan to deliver
* We plan to create a project that allows for the rendering of watercolor-esque images. The end result of this rendering program will generate images that have non-photorealistic watercolor effects.
* We plan to observe variations in watercolor style texture generation, where we could experiment with different methods/theory and compare the results. This may prove rather difficult, as how "watercolor-esque" an image is more subjective.

(2) what we hope to deliver
* If we are able to settle on a set "look" for watercolor style texture generation that isn't too subjective, we could look for optimizations and approximations to speed up the process. (showing graphs with speedup numbers).
* If possible, we could look to add another challenging aspect of watercolor rendering through reflections, as famous watercolor scenes contain reflections in water and lake surfaces. Our current project aims to add watercolor textures and lighting for mesh objects, but exploring reflections afterwards would be interesting if things go well.

## Schedule

## Resources
* [Non-Photorealistic Rendering
using Watercolor Inspired Textures and Illumination](https://www.dimap.ufrn.br/~motta/dim102/Projetos/NPR/Lume_PG01.pdf)
* [ART DIRECTED WATERCOLOR SHADER FOR NON-PHOTOREALISTIC
RENDERING WITH A FOCUS ON REFLECTIONS](https://core.ac.uk/download/pdf/154406433.pdf)
* We will be using code from projects 1 & 3 as a starting point for our renderer! We will use project 1's rasterization process and modifying the texture mapping process. We'll also use and modify project 3's lighting and shading methods.
