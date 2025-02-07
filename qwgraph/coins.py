import numpy as np
from math import pi 

"""
This is a global message !

"""

def generalized_coin(theta, phi, lbd):
	""" General coin

	$$
	U(\\theta,\\phi,\\lambda) = \\left(\\begin{matrix}
	\\cos \\theta & -e^{i\\lambda} \\sin \\theta \\\\
	e^{i\\phi} \\sin \\theta & e^{i(\\lambda+\\phi)} \\cos \\theta \\\\
	\\end{matrix}\\right)
	$$

	Args:
		theta (float): The angle $\\theta$.
		phi (float): The angle $\\phi$.
		lbd (float): The angle $\\lambda$.

	Returns:
		(complex numpy array): $U(\\theta,\\phi,\\lambda)$
	"""
	return np.array([[np.cos(theta) , -np.exp(1j*lbd)*np.sin(theta)],
		[np.exp(1j*phi)*np.sin(theta) , np.exp(1j*(lbd+phi))*np.cos(theta)]],dtype=complex)

def phase_shift(phi):
	""" Phase shift coin

	$$
	P(\\phi) = \\left(\\begin{matrix}
	1 & 0 \\\\
	0 & e^{i(\\phi)}\\\\
	\\end{matrix}\\right)
	$$

	Args:
		phi (float): The angle $\\phi$.

	Returns:
		(complex numpy array): $P(\\phi)$
	"""
	return np.array([
		[1,0],
		[0,np.exp(1j*phi)]],dtype=complex)

###############################################
##                 Pauli Gates               ##
###############################################
I = np.array([[1,0],[0,1]],dtype=complex)
""" 
$$
I = \\left(\\begin{matrix}
1 & 0 \\\\
0 & 1 \\\\
\\end{matrix}\\right)
$$
"""
X = np.array([[0,1],[1,0]],dtype=complex) 
""" 
$$
X = \\left(\\begin{matrix}
0 & 1 \\\\
1 & 0 \\\\
\\end{matrix}\\right)
$$
"""
Y = np.array([[0,-1j],[1j,0]],dtype=complex)
""" 
$$
Y = \\left(\\begin{matrix}
0 & -i \\\\
i & 0 \\\\
\\end{matrix}\\right)
$$
"""
Z = np.array([[1,0],[0,-1]],dtype=complex)
""" 
$$
Z = \\left(\\begin{matrix}
1 & 0 \\\\
0 & -1 \\\\
\\end{matrix}\\right)
$$
"""


###############################################
##                Common Gates               ##
###############################################
H = np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
""" 
$$
H = \\frac{1}{\\sqrt{2}}\\left(\\begin{matrix}
1 & 1 \\\\
1 & -1 \\\\
\\end{matrix}\\right)
$$
"""
S = np.array([[1,0],[0,1j]],dtype=complex)
""" 
$$
S = \\left(\\begin{matrix}
1 & 0 \\\\
0 & i \\\\
\\end{matrix}\\right)
$$
"""
T = np.array([[1,0],[0,np.exp(1j*pi/4)]],dtype=complex)
""" 
$$
T = \\left(\\begin{matrix}
1 & 0 \\\\
0 & e^{i\\frac{\\pi}{4}} \\\\
\\end{matrix}\\right)
$$
"""
Cx = np.array([[1,1j],[1j,1]],dtype=complex)/np.sqrt(2)
""" 
$$
C_x = \\frac{1}{\\sqrt{2}}\\left(\\begin{matrix}
1 & i \\\\
i & 1 \\\\
\\end{matrix}\\right)
$$
"""
Cy = np.array([[1,-1j],[-1j,1]],dtype=complex)/np.sqrt(2)
""" 
$$
C_y = \\frac{1}{\\sqrt{2}}\\left(\\begin{matrix}
1 & -i \\\\
-i & 1 \\\\
\\end{matrix}\\right)
$$
"""

