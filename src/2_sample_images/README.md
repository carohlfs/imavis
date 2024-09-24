# 2_sample_images

This folder contains three scripts that create the images included in Figures 2 and 3 of the paper.

The first script, print_test_case_svg.py prints the first test image from each dataset at different resolutions, as in Figure 2.

```bash
python3 ../src/2_sample_images/print_test_case_svg.py
```

The second script, get_bestworst.py, identifies the "best and worst" MNIST-style and RGB images that appear in Figure 3.

```bash
python3 ../src/2_sample_images/get_bestworst.py
```

The third script, print_bestworst_svg.py, prints those best & worst images from Figure 3. To ensure replicability, the index values for those best and worst images are hard-coded rather than taken from the get_bestworst.py file. If the probabilities from your model runs are the same, you should obtain the same index values in get_bestworst.py as are hardcoded in the print_bestworst_svg.py file.

```bash
python3 ../src/2_sample_images/print_bestworst_svg.py
```