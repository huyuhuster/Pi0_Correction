import math
from math import *


def definition1(R, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S):
	print("\n","*********  For the definition1: ","\n")
	dphi   = Phi_d - Phi_c
	dtheta = Theta_d - Theta_c
    # ****** angle of trapezoid
	angle_tor = atan(W1/L)
	angle_t1 = angle_tor + dtheta
	angle_t2 = angle_tor - dtheta
	angle_p1 = angle_tor + dphi
	angle_p2 = angle_tor - dphi
	print("Angle: ", angle_tor*180/3.1415, "   ", angle_t1*180/3.1415, "   ", angle_t2*180/3.1415)

    # ****** R of two threthod
	R1_th = sqrt(R*R + S*S - 2*S*R*cos(angle_t1))
	R2_th = sqrt(R*R + S*S - 2*S*R*cos(angle_t2))
	R1_phi = sqrt(R*R + S*S - 2*S*R*cos(angle_p1))
	R2_phi = sqrt(R*R + S*S - 2*S*R*cos(angle_p2))
	print("R = ", R1_th, "   ", R2_th)

    # ****** calculate the threthod 1 of position of gamma target  
    # R1 = 1250.
    # Theta1 = asin(R1/R1_th)
	Theta1 = Theta_c - acos((R*R+R1_th*R1_th-S*S)/(2*R*R1_th))
	Phi1 = Phi_c - acos((R*R+R1_phi*R1_phi-S*S)/(2*R*R1_phi))

    # ****** calculate the threthod 2 of position of gamma target  
    # R2 = R1 + W1*cos(Theta_d) 
    # Theta2 = asin(R2/R2_th)
	Theta2 = Theta_c + acos((R*R+R2_th*R2_th-S*S)/(2*R*R2_th))
	Phi2 = Phi_c + acos((R*R+R2_phi*R2_phi-S*S)/(2*R*R2_phi))
	
	print("deltatheta = ",  acos((R*R+R1_th*R1_th-S*S)/(2*R*R1_th)), "   ", acos((R*R+R2_th*R2_th-S*S)/(2*R*R2_th)))
	print("  Phi1 = ", Phi1, "  Theta1 =  ", Theta1)
	print("  Phi2 = ", Phi2, "  Theta2 =  ", Theta2)
	return Phi1, Phi2, Theta1, Theta2

 

def definition2(R, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S):
	print("\n", "*********  For the definition2: ", "\n")
	dphi   = Phi_d - Phi_c
	dtheta = Theta_d - Theta_c
	# angle of trapezoid
	angle_tor = 3.1415926/2
	angle_t1 = angle_tor + dtheta    # Theta
	angle_t2 = angle_tor - dtheta
	angle_p1 = angle_tor + dphi      # Phi
	angle_p2 = angle_tor - dphi
	print("Angle: ", angle_tor*180/3.1415, "   ", angle_t1*180/3.1415, "   ", angle_t2*180/3.1415)
	
	# R of two threthod
	R1_th = sqrt(R*R + Wc*Wc - 2*Wc*R*cos(angle_t1))
	R2_th = sqrt(R*R + Wc*Wc - 2*Wc*R*cos(angle_t2))
	R1_phi = sqrt(R*R + Wc*Wc - 2*Wc*R*cos(angle_p1))
	R2_phi = sqrt(R*R + Wc*Wc - 2*Wc*R*cos(angle_p2))
	print("R = ", R1_th, "   ", R2_th)

	
	print("deltatheta = ",  acos((R*R+R1_th*R1_th-Wc*Wc)/(2*R*R1_th)), "   ", acos((R*R+R2_th*R2_th-Wc*Wc)/(2*R*R2_th)))
	# calculate the threthod 1 of position of gamma target  
	Theta1 = Theta_c - acos((R*R+R1_th*R1_th-Wc*Wc)/(2*R*R1_th))
	Phi1 = Phi_c - acos((R*R+R1_phi*R1_phi-Wc*Wc)/(2*R*R1_phi))
	
	# calculate the threthod 2 of position of gamma target  
	Theta2 = Theta_c + acos((R*R+R2_th*R2_th-Wc*Wc)/(2*R*R2_th))
	Phi2 = Phi_c + acos((R*R+R2_phi*R2_phi-Wc*Wc)/(2*R*R2_phi))
	print("Phi1 = ", Phi1, "  Theta1 =  ", Theta1)
	print("Phi2 = ", Phi2, "  Theta2 =  ", Theta2)
	return Phi1, Phi2, Theta1, Theta2


def definition3(R, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S):
	print("\n", "*********  For the definition3: ", "\n")
	dphi   = Phi_d - Phi_c
	dtheta = Theta_d - Theta_c
	# angle of trapezoid
	angle_tor = 3.1415926/2
	angle_t1 = dtheta    # Theta
	angle_t2 = atan(L*2/(W2-W1))
	angle_p1 = dphi      # Phi
	angle_p2 = atan(L*2/(W2-W1))
	print("Angle: ", angle_tor, "   ", angle_t1, "   ", angle_t2)
	print("Angle: ", angle_tor*180/3.1415926, "   ", angle_t1*180/3.1415926, "   ", angle_t2*180/3.1415926)
	
	# lenghth
	at1 = Wc*sin(3.1415926-angle_t2-angle_t1)/sin(angle_t2)   # theta
	at2 = Wc*sin(angle_t2-angle_t1)/sin(3.1415926-angle_t2)
	ap1 = Wc*sin(3.1415926-angle_p2-angle_p1)/sin(angle_p2)   # phi
	ap2 = Wc*sin(angle_p2-angle_p1)/sin(3.1415926-angle_p2)
	
	
	print("at1,2: ", at1, "  ", at2, '\n', "ap1,2: ", ap1, "   ", ap2)
	
	
	print("deltatheta = ",  atan(at1/R), "   ", atan(at2/R))
	# calculate the threthod 1 of position of gamma target  
	Theta1 = Theta_c - atan(at1/R)
	Phi1 = Phi_c - atan(ap1/R)
	
	# calculate the threthod 2 of position of gamma target  
	Theta2 = Theta_c + atan(at2/R)
	Phi2 = Phi_c + atan(ap2/R)
	print("Phi1 = ", Phi1, "  Theta1 =  ", Theta1)
	print("Phi2 = ", Phi2, "  Theta2 =  ", Theta2)
	return Phi1, Phi2, Theta1, Theta2
	


def definition4(R, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S):
	print("\n", "*********  For the definition4: ", "\n")
	# angle of trapezoid
	angle_tor = 3.1415926/2
	angle_t1 = angle_tor-Theta_d    # Theta
	angle_t2 = atan(L*2/(W2-W1))
	angle_p1 = dphi      # Phi
	angle_p2 = atan(L*2/(W2-W1))
	
	# lenghth
	at1 = Wc*sin(angle_t2-angle_t1)/sin(3.1415926-angle_t2)
	at2 = Wc*sin(3.1415926-angle_t2-angle_t1)/sin(angle_t2)   # theta
	ap1 = Wc*sin(angle_p2-angle_p1)/sin(3.1415926-angle_p2)
	ap2 = Wc*sin(3.1415926-angle_p2-angle_p1)/sin(angle_p2)   # phi
	
	# R
	R1_th = sqrt(R*R + at1*at1 - 2*R*at1*cos(3.1415926-Theta_c))
	R2_th = sqrt(R*R + at2*at2 - 2*R*at2*cos(Theta_c))
	
	print("Angle: ", angle_tor*180/3.1415, "   ", angle_t1*180/3.1415, "   ", angle_t2*180/3.1415)
	
	# R of two threthod
	
	# calculate the threthod 1 of position of gamma target  
	Theta1 = Theta_c - acos((R*R+R1_th*R1_th-at1*at1)/(2*R*R1_th))
	Phi1 = Phi_c - 0.0218175
	
	# calculate the threthod 2 of position of gamma target  
	Theta2 = Theta_c + acos((R*R+R2_th*R2_th-at2*at2)/(2*R*R2_th))
	Phi2 = Phi_c + 0.0218175
	
	
	print("Phi1 = ", Phi1, "  Theta1 =  ", Theta1)
	print("Phi2 = ", Phi2, "  Theta2 =  ", Theta2)
	return Phi1, Phi2, Theta1, Theta2
	

 

def getCrystalPhiAndTheta(crystal_ID, definition):
    id_theta_c = {}
    id_phi_c = {}
    id_R_c = {}
    id_theta_d = {}
    id_phi_d = {}

    file_pos = open("digits_position_dirction/digits_position.txt")
    line = file_pos.readline()
    line = file_pos.readline()
    while line:
        line_list = line.rstrip('\n').split('\t')
        id_theta_c[line_list[0]]= float(line_list[1])
        id_phi_c[line_list[0]] = float(line_list[2])
        id_R_c[line_list[0]] = float(line_list[6])
        line = file_pos.readline()
    file_pos.close()

    file_dir = open("digits_position_dirction/digits_direction.txt")
    line = file_dir.readline()
    line = file_dir.readline()
    while line:
        line_list = line.rstrip('\n').split('\t')
        id_theta_d[line_list[0]] = float(line_list[1])
        id_phi_d[line_list[0]]   = float(line_list[2])
        line = file_dir.readline()
    file_dir.close()

    # ****** position of crystal  
    R_c     = id_R_c[crystal_ID]*10.
    Phi_c   = id_phi_c[crystal_ID]
    Theta_c = id_theta_c[crystal_ID]
    
    # ****** direction of crystal
    Phi_d   = id_phi_d[crystal_ID]
    Theta_d = id_theta_d[crystal_ID]


    dphi   = Phi_d - Phi_c
    dtheta = Theta_d - Theta_c
    # size of crystal
    L = 300.
    W1 = 55.
    W2 = 60.       
    Wc = (W1+W2)/4
    S = 0.5*sqrt(L*L+W1*W1)

    print("Crystal ", crystal_ID, ",  has Theta = ", Theta_c, " and Phi =  ", Phi_c)
    print("delta Phi = ", dphi, ",  deta Theta = ", dtheta)
    if definition=='1':
        Phi1, Phi2, Theta1, Theta2 = definition1(R_c, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S)
    elif definition=='2':
        Phi1, Phi2, Theta1, Theta2 = definition2(R_c, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S)
    elif definition=='3':
        Phi1, Phi2, Theta1, Theta2 = definition3(R_c, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S)
    elif definition=='4':
        Phi1, Phi2, Theta1, Theta2 = definition4(R_c, Phi_c, Theta_c, Phi_d, Theta_d, L, W1, W2, Wc, S)
    else:
        print("Wrong definition: ", definition, " !!!")

    return Phi1, Phi2, Phi_c, Theta1, Theta2, Theta_c

#getCrystalPhiAndTheta('2626', '2')

			
