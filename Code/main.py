import os

# Set the working directory
os.chdir('Code')
# Call Video_Stabilization.py
os.system('python Video_Stabilization.py')
# Call background subtraction
os.system('python Background_Subtraction.py')
# Call Matting
os.system('python Matting.py')
# Call Tracking
os.system('python Tracking.py')
