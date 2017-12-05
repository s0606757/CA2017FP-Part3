# NCTU 2017 Computer Architecture Final Project - Part 3

**Part-3：Speedup irregular sparse convolution.**

## Download
At command line type:
<pre>
git clone https://github.com/s0606757/CA2017FP-Part3.git
</pre>

## Three sub-directory

### ./data
This directory contains the input data for the base program：
* ./data/filter.txt storing irregular dense filters (sparsity=78%)
* ./data/neuron.txt storing irregular dense neuron (sparsity=78%)
* ./data/filter_COO_irregular.txt storing irregular sparse filters (sparsity=78%)
* ./data/neuron_COO_ irregular.txt storing irregular sparse neuron (sparsity=78%)
* ./data/filter_COO_regular.txt storing sparse filters (sparsity=78%)
* ./data/neuron_COO_ regular.txt storing sparse neuron (sparsity=78%)

* TA will test your code with filter_COO_irregular.txt and neuron_COO_ irregular.txt (or you can use other sparse format and please compress the data into zip file)


### ./device
The program under this directory can show the device information.
#### usage
<pre>
cd ./device
make
make run
</pre>

### ./example
There is one example(InnerProduct) under this directory.
#### usage
<pre>
cd ./example/InnerProduct/
make
make run
</pre>

## Usage of the base program
<pre>
make
make run
</pre>

## Evaluation
We will compare the execution time to get the speedup by
<pre>
Speedup = convLayerGPU_TA_execTime  / convLayerGPU_Your_execTime
PS：convLayerGPU_XXX_execTime includes : only execution time.
</pre>

## Grading Policy
**(A) Completeness (20%)**<br/>
&nbsp;    Your result(convLayerGPU_Your) must be correct (Pass the check) (10%)<br/>
&nbsp;&nbsp;&nbsp;    Your design(convLayerGPU_Your) is faster than convLayerGPU_TA (10%)<br/>
**(B) Report (40%)**<br/>
&nbsp;     1.	Algorithm (10%) <br/>
&nbsp;&nbsp;&nbsp;    2.	How do you (10%)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       - increase data reuse<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      - reduce branch divergence or increase memory coalescing<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      - implement other optimization <br/>
&nbsp;&nbsp;    3.	Comparing part 3 with part 2 , do you get speedup? why or why not?(10%)<br/>
&nbsp;    4.	Show how you use NVVP to help you find and solve perf(5%)<br/>
&nbsp;    5.	Feedback(5%)<br/>

**(C) Performance Rank (30%)**<br/>
&nbsp;&nbsp;&nbsp;    We will rank your CUDA kernels’ performance(execution time) on GTX K20C<br/>
&nbsp;&nbsp;&nbsp;    The fastest one will get 30% and the last one will get 1%<br/>

## Other Rules
* It’s a team work, 1 ~ 6 people in one team <br/>
   - Register (**Please reedit!**)[here](https://docs.google.com/spreadsheets/d/1pTu70p91DqzbtLaKE4OWdUmqs7FS-8vAtqs0gH41xcY/edit?usp=sharing) before deadline.<br/>
* [Account list](https://docs.google.com/spreadsheets/d/1hLfJjv58QsXRwLlma45IflcpicqlQFgYiKp77vlJokk/edit#gid=0)
* Compress following files into one zip file **LeaderID_FP3.zip** **(ex：066066_FP3.zip)** and upload to E3.<br/>
* (a) LeaderID_Report_FP3.pdf <br/>
* (b) CNNConvLayer.cu and CNNConvLayer.h <br/>
* (c) Makefile and ./data  <br/>
* One team only need to upload **one** package to E3.<br/>
* Make sure TA can compile and run your code with “make” and “make run” on the provided server.<br/>
* **Any CUDA library is forbidden to use in this project !!!** <br/>
* **DELAY IS NOT ACCEPTABLE !!!** <br/>
* **Due day：2017/01/16(Tue) 23:50** <br/>

## Useful Reference
* Neural Network Pruning [Here](https://arxiv.org/pdf/1506.02626.pdf)
* Sparsity in Neurons [Here](http://www.ece.ubc.ca/~aamodt/papers/Cnvlutin.ISCA2016.pdf)
* Sparse Foramat in Matrix Multiplication [Here](https://pdfs.semanticscholar.org/9abb/086fabdcd2853ed8303c0f9a62cf4b917a62.pdf)
* Implement Sparse Matrix Multiplication with CUDA [Here](http://wnbell.com/media/2008-12-NVR-SpMV/nvr-2008-004.pdf)
* An Accelerator for Sparse CNN [Here](http://people.csail.mit.edu/anurag_m/papers/2017.scnn.isca.pdf)
* Sparse Format - Nvidia [Here](https://drive.google.com/file/d/0B-mvsV4UBCFFbEhpMzFIbUVLVGs/view?usp=sharing )


