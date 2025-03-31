I have learned about Radix Sort today, Radix sort is a sorting algorithm that sorts the number according to the bucket for n bits in a number.
the process includes:
1. we seperate the keys based on least significant bit
2. then the second least significant bit
3. go on till the most signicant bit
4. Do exclusive scan
5. Next find the destination of each 0 and 1 using numbers from exclusive scan, 
How to Find the Destination Index of Each Element

🟦 For a zero (0):
	•	Destination index = number of zeros to the left
	•	= number of elements to the left − number of ones to the left
	•	= element index − number of ones to the left

⸻

🟩 For a one (1):
	•	Destination index = number of zeros in total + number of ones to the left
	•	= (number of elements in total − number of ones in total) + number of ones to the left
	•	= input size − number of ones in total + number of ones to the left

⸻

✅ What do we need to compute?
	•	Number of ones to the left of each element
→ Use an exclusive scan to compute this.

Now wait, isn't this inefficient so now? We are going to use shared memory!
 1. We are going to divide the input into blocks and and we are going to have each thread for each element in the block then we are going to use
 shared memory to sort the elements according to 0's and 1's like we did in step 5 above.
 2. We are going to place it into destination of its global array.
 3. And we are going to repeat this for n bits.  


