import numpy as np

def arctanMap(x):

	return (2/np.pi)*(np.arctan( (np.pi/2) * (x ) )) + 1

def invArctanMap(y):

	return (2/np.pi)*(np.tan( (np.pi/2)* (y-1)))

def test_inversion():
	xs = np.linspace(-2,2,100)
	ys = [arctanMap(x) for x in xs]
	nxs = [invArctanMap(y) for y in ys]

	for nx, x in zip(nxs, xs):
		assert(abs(x-nx)<1e-6)



def dArctanMap(x):

	return 1/(1 + ((np.pi/2) * x)**2)

if __name__ == '__main__':
	test_inversion()