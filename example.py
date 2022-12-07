
import funAD as ad

def f(x):
    ad.sin(ad.exp(x)) + 6 ** (ad.cos(x)) / x - 2 * x

# df is now a function that takes in x value 
#(single float number) and output its derivative at given x value (single float number)
df = ad.function(f) 

for x in range(5):
    der = df(x)  # der is a single float number
    val = f(x)  # val is a single float number