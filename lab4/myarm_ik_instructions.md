For MyArm IK exercise:



* You need pinocchio and meshcat libraries. Run:


**pip3 install pin**

**pip3 install meshcat**

**pip3 install meshcat-shapes**



* Put warm start for the initial configuration if you want, changing the proper variable to true:



**warm\_start = False  # if True it finds good initial configurations for the IK search, if False it sets q0 = 0**



* Change the solver with the parameter ik\_solver in the code line:



**ik\_solver = "dls\_qp"  # "newton", "dls\_lm", "dls\_qp"**



* When comparing, remove the twist constraints from the QP solver, in order to perform a fair comparison with the other two methods.



* Run the python codes try\_fullpose\_IK.py and try\_position\_IK.py from "myarm\_ik" folder, with:



**python3 -m try\_fullpose\_IK**



and



**python3 -m try\_position\_IK**



