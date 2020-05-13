# 184 Final Project Report: Watercolor-esque Non-Photorealistic Rendering

## Abstract
This project aims for non-photorealistic rendering of watercolor-esque images. More specifically, we implemented algorithms from Curtis 97’s paper (see References) on generative rendering of watercolor onto 2D spaces as a pre processing simulation. However, our project actually applies Curtis’s ideas in 3D rendering, and implements watercolor style coloring for meshes in the 3D image space. This project utilizes project 3-1 coding for ray tracing as a base. It then utilizes concepts from the Curtis paper to allow for physical fluid simulation and also a pigment based lighting model for water color shading. 

## Technical Approach
We first started by looking at the papers that we referenced in the project proposal but eventually settled on following the writeup from the Curtis 97 paper. The primary difference with the Curtis paper is that it involved a physical simulation of the watercolors, whereas others were done using post-processing techniques such as shaders and mathematical color transformations.

The project had two main components -- the watercolor fluid simulation and the renderer. The goal of the renderer was to generate water-color-esque patches of pigment on any object mesh. The goal for the simulation is to create a 3-layer fluid simulation that accounts for pigment movement across mesh cells in water as well as absorption into and desorption from the “paper” mesh. The fluid simulation can be broken down into three layers - the shallow-water layer, the pigment-deposition layer, and the capillary layer. The shallow water layer is where the water and pigment flow above the surface of the paper, the pigment-deposition layer where the pigment is adsorbed into and desorbed from the paper, and the capillary layer is where the water is absorbed into the paper and diffused by capillary action (creating the back run effect).

For the renderer, our approach was to modify the ray-traced renderer from project 3-1 to allow different faces of a mesh to have different reflectance spectrums. This was done by modifying faces of the HalfedgeMesh in the simulation phase to hold a desired reflectance spectrum (set at the end of simulation), then propagating that spectrum into the primitives used by the renderer. Finally, we modified the raytracing code so the BSDF of “watercolored” meshes would use that custom reflectance. 

For the physical simulation, we closely followed the paper’s 3-layer fluid simulation algorithms, but modified them to match our 3D renderer instead of the paper’s 2D post-processing watercolorization. Specifically, the biggest challenge was simulating paper - the Curtis 97 simulation set the entire 2D scene as a whole 2D grid of paper cells with randomly generated height for “roughness”. Instead, we used the faces of the mesh object as our paper cells. As such, we had to adapt the Curtis 97 fluid simulation of pigments flowing in 4 directions for each paper cell to just 3 directions for each face’s 3 neighbors. We also chose to generate the height variation through the face intersection angles to other neighboring faces. Intuitively, a face F whose neighboring faces were tilted towards F’s normal would have a smaller height, while a face F2 with neighbors pointing away from F2’s normal would have a larger height. This technique produces heights that are highly dependent on local context; future work might include incorporating convexity information from a wider set of faces than just immediate neighbors. Height gradients were created via local differences of heights.

In our implementation, we run everything in a simulate_mesh function that takes in a mesh object. We run four main steps at each time step: MoveWater, MovePigment, TransferPigment, and SimulateCapillaryFlow. We start by generating water color patches (collections of mesh Faces that we want to watercolor) and then instantiating them with all the parameters necessary for the simulation. In MoveWater, we update velocities in each direction for each cell to satisfy the 6 conditions for shallow water described in (Curtis 97). The implementation in Curtis relied on a discretization of the shallow water equations based on their grid structure that was then solved using Euler’s method. The first step of water movement involved updating water velocities across cell boundaries based on surrounding water velocities and cell water pressures. We stored a non-negative outwards velocity in each halfedge, then updated those velocities using a local coordinate system for each half edge that allowed us to calculate analogs of the “horizontal” and “vertical” velocities of surrounding edges and cells. In MovePigment, we followed the paper by distributing pigments from each face to its neighbors according to the rate of fluid movement out of the cell in each direction. In TransferPigment, we adjust the amount of pigment on the paper by calculating how much pigment is absorbed into the paper and desorbed out of the paper through the pigment density, staining power, and granulation properties (also as done in Curtis 97). The Curtis 97 simulation also had a simulateCapillaryFlow function that generated the backrun effect of diffusing paper through paper properties of pressure and saturation, but we elected not to do so since we already predefined watercolor patches before the simulation. Instead, we elected to create a dry-brush effect, where a brush that is almost dry only applies paint to the rougher areas on the paper.

To properly display pigment colors, the Curtis 97 paper utilized the Kubelka-Munk model to perform the optical compositing of pigment layers, with each pigment having absorption and scattering coefficients. The model then uses those coefficients to compute and display reflectance and transmittance. We implemented the Kubelka-Munk model’s compositing of pigments, and actually rendered several colors with parameters from (Curtis 97). We were concerned that the watercolor effects were ignoring the underlying mesh color, so we added in an implicit layer of "mesh colored" paint into every color-composition we rendered. This tended to yield much better results.

### Issues Faced

One significant problem we ran into was similar but it was on the calculation of realistic velocities and flow. In the Curtis 97 simulation, the velocities were solved forward using Euler’s method in a 2D square grid.  Working with 3D meshes with faces as paper cells, we were unsure whether our adaptation of their discretization to 3D meshes was mathematically valid. Because we didn’t implement capillary flow, we also did not take into account saturation or capacity of the “paper” (mesh). Another issue was getting the pigment rendering to simply appear more watercolor-esque. Due to rendering our 3D mesh objects under lighting and shading, it was difficult to see the differences in watercoloring - the renders that worked best were under good lighting conditions. We also ran into a slight issue where extra simulation steps tended to cause “spotty” concentration of watercolor, which we were unable to fully resolve. We also ended up having to do much more math than anticipated to get the angle between adjacent faces (a simple inverse cosine of normal vector dot product doesn’t work because cos(x) = cos(-x) and we care about the sign of x)

### Lessons Learned
The main lesson we learned is that a paper writeup is not as simple as 1-1 with an implementation, especially when there are more components involved. We originally presumed that this project would be mostly about implementing what we saw in the Curtis 97 paper, but later realized that it became much more difficult - namely because we were rendering in 3D with lighting involved. As such, we should have more carefully planned out our project and then gradually add features, instead of directly writing code without giving thought to the overall implementation details. Lastly, we all learned to appreciate simulation renderers much more - there is much more to physical watercolor fluid simulations than we expected and it was very rewarding to tackle it with our own ideas. We learned a great deal more about optical compositing to display color, physical paper and fluid properties, and special watercolor features.

## Results
bunny rendered with 10 timesteps of simulation
![bunny rendered with 10 timesteps of simulation](/docs/images/bunny_10.png)

bunny rendered with 100 timesteps of simulation
![bunny rendered with 100 timesteps of simulation](/docs/images/bunny_100.png)

bunny rendered with 1000 timesteps of simulation
![bunny rendered with 1000 timesteps of simulation](/docs/images/bunny_1000.png)

In images with more simulation timesteps, the final rendering appears "drier", therefore making it look more like true watercolor.
However, it can sometimes be hard to see watercoloring without strong illumination. The following images of dragons look more watercolor-esque because the scene is well-illuminated.

![dragon rendered with 100 timesteps of simulation](/docs/images/dragon_64_32_colorful.png)

The dragon below shows the effects of paints overlapping when using the Kubelka-Munk model; the purple and yellow paints that overlap create an orange color.

![dragon rendered with 100 timesteps of simulation](/docs/images/dragon_64_32_overlap_100.png)

The dragon below is a poor result that we got when we didn't try to account for the underlying color of the mesh. It turned out that including the underlying mesh color as an implicit "layer of paint" was crucial to generating the above images so they did not end up looking extremely saturated with paint like the image below.

![dragon rendered with 100 timesteps of simulation](184final/docs/images/dragon_64_32_no_meshbias_100.png)

## Video
https://youtu.be/FOn81lEtZzE

## References
* [Non-Photorealistic Rendering
using Watercolor Inspired Textures and Illumination](https://www.dimap.ufrn.br/~motta/dim102/Projetos/NPR/Lume_PG01.pdf)
* [ART DIRECTED WATERCOLOR SHADER FOR NON-PHOTOREALISTIC
RENDERING WITH A FOCUS ON REFLECTIONS](https://core.ac.uk/download/pdf/154406433.pdf)
* We used code from project 3.1 as a starting point for our renderer.
* [Computer-Generated Watercolor](https://www.cs.princeton.edu/courses/archive/fall00/cs597b/papers/curtis97.pdf)

## Contributions
* Andrew: Mainly focused on implementing and re-designing fluid simulation with algorithms from the Curtis 97 reference paper, and implemented extra features such as dry-brush effect.
* Evan: Modified existing project 3-1 code to render water colors instead. Designed height function and mesh-specific adaptations to the water movement simulation functions.
* Johnathan: Mainly focused on implementing fluid simulation with algorithms from the Curtis 97 reference paper, and tweaking parameters to find optimal simulation values for most watercolor-esque results.
