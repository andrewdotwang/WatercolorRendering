# 184 Final Project Proposal: Watercolor-esque Non-Photorealistic Rendering

## Summary & Team members
Team members: Andrew Wang, Evan Lohn, Johnathan Zhou  

Our final project idea is in the realm of non-photorealistic rendering. More specifically, we plan to utilize several layers of textures along with lighting to mimic & render the apperance of watercolor. 

## Problem Description

As more modern art move to digital forms, computer graphic rendering techniques to mimic traditional hand drawn art forms are becoming popular. One such art form is watercolors, where artists draw charming images through the blending of rich and vibrant colors. Our group was especially interested in the non-photorealistic rendering of watercolor-esque images. This marks a contrast from the typical photorealism focus in traditional computer graphics, allowing for more artistic flexibility and expression. However, this project has a big challenge in texture generation, where creating a new watercolor style texture for each mesh in the 3D image space could be difficult. To implement watercolor-esque non-photorealistic rendering, we will follow existing papers on the subject and utilize similar techinques. More specifically, we will (1) use existing project 1 code for rasterization and project 3 coding for lighting (2) introduce watercolor style texture generation and mapping to simulate brush wash like textures (3) create a pigment based lighting model to allow for shading with the set of colors like those found in typical watercolor paintings.

## Goals and Deliverables

What we plan to deliver
* We plan to create a project that allows for the rendering of watercolor-esque images. The end result of this rendering program will generate images that have non-photorealistic watercolor effects.
* We plan to observe variations in watercolor style texture generation, where we could experiment with different methods/theory and compare the results. This may prove rather difficult, as how "watercolor-esque" an image is more subjective.

What we hope to deliver:
* If we are able to settle on a set "look" for watercolor style texture generation that isn't too subjective, we could look for optimizations and approximations to speed up the process. (showing graphs with speedup numbers).
* If possible, we could look to add another challenging aspect of watercolor rendering through reflections, as famous watercolor scenes contain reflections in water and lake surfaces. Our current project aims to add watercolor textures and lighting for mesh objects, but exploring reflections afterwards would be interesting if things go well.

## Schedule

* Week 1: Research on existing papers and familiarize with the underlying techniques and theory.
* Week 2: Start writing code, playing around with different implementations. Experiment with different methods to see variations in watercolor style texture generation that we like the most.
* Week 3: Finalize the implementation of features we plan to deliver listed in the Goals and Deliverables section. Ensure that there are no outstanding flaws and bugs. Look into extra features if possible.
* Week 4: Work on extra features listed in the Goals and Deliverables section if possible. Finalize the project presentation.

## Resources
* [Non-Photorealistic Rendering
using Watercolor Inspired Textures and Illumination](https://www.dimap.ufrn.br/~motta/dim102/Projetos/NPR/Lume_PG01.pdf)
* [ART DIRECTED WATERCOLOR SHADER FOR NON-PHOTOREALISTIC
RENDERING WITH A FOCUS ON REFLECTIONS](https://core.ac.uk/download/pdf/154406433.pdf)
* We will be using code from projects 1 & 3 as a starting point for our renderer! We will use project 1's rasterization process and modifying the texture mapping process. We'll also use and modify project 3's lighting and shading methods.


# 184 Final Project Report: Watercolor-esque Non-Photorealistic Rendering

## Abstract
This project aims to allow for non-photorealistic rendering of watercolor-esque images. More specifically, we implemented algorithms from Curtis 97’s paper on generative rendering of watercolor onto 2D spaces as a post processing simulation. However, our project actually applies Curtis’s ideas in 3D rendering, and implements watercolor style coloring for each mesh in the 3D image space. This project utilizes existing project 1 code for rasterization and project 3 coding for lighting as a base. It then utilizes concepts from the Curtis paper to allow for physical fluid simulation and also a pigment based lighting model for water color shading. 

We first started by looking at the papers that we referenced in the project proposal but eventually settled on following the writeup from the Curtis 97 paper. The primary difference with the Curtis paper is that it is more of a physical simulation of the watercolors compared to the others, which were done using post-processing techniques such as shaders and mathematical color transformations.

## Technical Approach
Overall, we implemented 3 main components -- the renderer, the physics of fluid simulation, and properly displaying color.

For the renderer, our general approach was to __. 

The fluid simulation can be broken down into three layers - the shallow-water layer, the pigment-deposition layer, and the capillary layer. The shallow water layer is where the water and pigment flow above the surface of the paper, the pigment-deposition layer where the pigment is adsorbed and desorbed from the paper, and the capillary layer where the water is absorbed into the paper and diffused by capillary action (creating the back run effect). We closely followed the paper’s 3-layer fluid simulation algorithms, but modified them to match our 3D renderer instead of the paper’s 2D post-processing watercolorization. Specifically, the biggest challenge was simulating paper - the Curtis 97 simulation set the entire 2D scene as a whole 2D grid of paper cells with randomly generated height for “roughness”. Instead, we used the faces of the mesh object as our paper cells, and generated the height variation through the face intersection angles to other neighboring faces. As such, we encountered a big problem of adapting the Curtis 97 fluid simulation of pigments flowing in 4 directions for each paper cell to just 3 directions for each face’s 3 neighbors.

In our implementation, we run everything in a simulate_mesh function that takes in a mesh object. We run four main steps at each time step: MoveWater, MovePigment, TransferPigment, and SimulateCapillaryFlow. We first start by generating water color patches, lists of Faces (paper cells) that we want to perform watercolor rendering and then instantiating them with all the parameters necessary for the simulation. In MoveWater, we update velocities in each direction for each cell to slow down fluid and pigment flow over time. While the Curtis 97 paper updated velocities by Euler's method, we scale velocities in each direction through the height gradients of each face to its neighbors, as fluid and pigments will flow faster from peaks to valleys. In MovePigment, we mainly followed the paper by distributing pigments from each face (cells) to its neighbors according to the rate of fluid movement out of the cell in each direction. In TransferPigment, we adjust the amount of pigment on the paper by calculating how much pigment is absorbed into the paper and desorbed out of the paper through the pigment density, staining power, and granulation properties. The Curtis 97 simulation also had a simulateCapillaryFlow function that generated the backrun effect of diffusing paper through paper properties of pressure and saturation, but we elected not to do so since we already predefined watercolor patches before the simulation. Instead, we elected to create a dry-brush effect, where a brush that is almost dry only applies paint to the rougher areas on the paper.

To properly display pigment colors, the Curtis 97 paper utilized the Kubelka-Munk model to perform the optical compositing of pigment layers, with each pigment having absorption and scattering coefficients. The model then uses those coefficients to compute and display reflectance and transmittance. However, as we were working with 3D meshes under lighting and shading instead of post processing 2D surfaces, we ran into trouble to account for these optical properties. As such, we simply used the pigment concentrations in each face (cell) and scaled them by their RGB value to generate the final RGB color.

One significant problem we ran into was similar but it was on the calculation of realistic velocities and flow. In the Curtis 97 simulation, the velocities were solved forward using Euler’s method in a 2D square grid.  Working with 3D meshes with faces as paper cells, we were unable to update velocities based on important qualities of watercolor on paper, such as pressure, saturation, and capacity of paper. Instead, we redefined our velocity updating function to depend more on the height gradients of faces and neighboring faces to account for the roughness of paper and less on those other properties. Another issue was getting the pigment rendering to simply appear more watercolor-esque. Due to rendering our 3D mesh objects under lighting and shading, it was difficult to see the differences in watercoloring - we were unable to calculate reflectance properly with the pigment absorption and scattering properties. Instead, we calculated final RGB values by simply scaling the pigment concentrations by their RGB value, and thickened pigment concentrations to better see the variation in color from pigment flow.

The main lesson we learned is that it is not easy to have a simple 1-1 implentation from the reference paper, especially when there are more components involved. We originally presumed that this project would be mostly about implementing what we saw in the Curtis 97 paper, but later realized that it became much more difficult - namely because we were rendering in 3D with lighting involved. As such, we should have more carefully planned out our project and then gradually add extra features, instead of directly writing code without giving thought to the overall implementation details. Lastly, we all learned to appreciate simulation renderers much more - there is much more to physical watercolor fluid simulations than we expected and it was very rewarding to tackle it with our own ideas. It is eye opening to note that there are many different ways to approach the same basic idea of watercolor-esque non-photorealistic rendering. We learned a great deal more about optical compositing to display color, physical paper and fluid properties, and special watercolor features.
