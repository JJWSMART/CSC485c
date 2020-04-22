Create Running Graph :
  
       10.txt
       1000.txt
       3000.txt
       3500.txt

Command: g++ -std=c++17 data_generator.cpp -o d

         ./d 10.txt > output.txt

Compare Result:

         Diff -s file1.txt file2.txt


Command run file: nvcc filename.cu -o exe

                  ./exe file1.txt