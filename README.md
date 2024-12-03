# Light field rendering with tensor cores
The program in this repository inteprolates several light field images at once using tensor cores available in NVIDIA RTX GPUs using their Cuda API.   


Example (use -h for the description):
```
./lfInterpolator -i ./input/ -o result/ -t 0.0,0.0,1.0,1.0 -a 1.783 -m TEN_WM -f 0.23
```
