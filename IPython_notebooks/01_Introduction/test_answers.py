def check_x(x_variable):
    if type(x_variable)==float:
        print('Your x-value is a float! Great job!')
    else:
        print('Your x-value is not a float. Try again!')

def check_y(y_variable):
    if type(y_variable)==int:
        print('Your y-value is an int! Great job!')
    else:
        print('Your y-value is not an int. Try again!')

def check_z(x_variable):
    if type(x_variable)==str:
        print('Your z-value is a string! Great job!')
    else:
        print('Your z-value is not a string. Try again!')



def check_math_operation(add, subtract, multiply, divide, exponent, modulus):
    if add == (1745+8765.1):
        print('Your variable ADD is correct! Wonderful')
    else:
        print('Your variable ADD is incorrect! Try again.')


    if subtract == (4587.6-423.9):
        print('Your variable SUBTRACT is correct! Wonderful')
    else:
        print('Your variable SUBTRACT is incorrect! Try again.')


    if multiply == (54*76):
        print('Your variable MULTIPLY is correct! Wonderful')
    else:
        print('Your variable MULTIPLY is incorrect! Try again.')

    if divide == (76/9):
        print('Your variable DIVIDE is correct! Wonderful')
    else:
        print('Your variable DIVIDE is incorrect! Try again.')

    if exponent == (37**6):
        print('Your variable EXPONENT is correct! Wonderful')
    else:
        print('Your variable EXPONENT is incorrect! Try again.')
    if modulus == (87%6.9):
        print('Your variable MODULUS is correct! Wonderful')
    else:
        print('Your variable MODULUS is incorrect! Try again.')

def check_parentheses(answer):
    if answer == (8+(9**(2+5)))/(6-(5*7)):
        print('Your answer is correct! Wonderful')
    else:
        print('Your answer is incorrect! Try again.')

def check_function(answer):
    if answer == (3*(3-5))/(5**3):
        print('Your answer is correct! Wonderful')
    else:
        print('Your answer is incorrect! Try again.')





def check_forloop1(start,end,step_size,value):
    if start==10:
        print('Your start variable is correct! Great job!')
    else:
        print('Your start variable is incorrect. Try again!')
    if end==0:
        print('Your end variable is correct! Great job!')
    else:
        print('Your end variable is incorrect. Try again!')
    if step_size==-2:
        print('Your step_size variable is correct! Great job!')
    else:
        print('Your step_size variable is incorrect! Try again!')
    if value==36:
        print('Your value variable is correct! Great job!')
    else:
        print('Your value variable is incorrect! Try again!')
#
# def check_forloop2(variable):
#     x_temp=0
#     for i in range(21):
#         x_temp += (x_temp + 31)/94
#     if variable==x_temp:
#         print('You are correct! Great job!')
#     else:
#         print('You are incorrect. Try again!')

# def check_fuelcells(fcn):
#     if fcn==8:
#         print('Your fuel cell calculation is correct! Great job!')
#     else:
#         print('Your fuel cell calculation is incorrect! Try again!')
