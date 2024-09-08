Simpmle CUDA exercises, meant for complete beginners. 


- Explore the GPU in your system by looking at its device properties.
- Write your first own Hello World CUDA kernel and understand the hardware mapping of threads.
- Work on vector addition and matrix multiplication examples to become more familiar with CUDA.

Pre-requisites:
- Access to a server containing an Nvidia GPU and with a CUDA installation
 
(For people with a CERN account, it is possible to interactively use GPUs on the grid, as desribed [here](https://batchdocs.web.cern.ch/tutorial/exercise10.html).)


Clone the repository into a directory on the server containing the Nvidia GPU. 
Follow the instructions in the pdf document and use the code snippets in the `exercises` folder to start from. Note that you should keep the file hierarchy as it is during start-up. All examples 
make use of a helper function defined within helpers.h, so that file has to be in the same directory as the source code you are compiling.

If you want to check your implementation, you can compare to the solution in the `solutions` folder, but only once you are done ;-)