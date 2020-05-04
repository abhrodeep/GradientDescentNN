import numpy as np

#4 input OR gate  [input,input,input,input,bias]
inputsys=np.array([[0,0,0,0,0.8],[0,0,0,1,0.8],[0,0,1,0,0.8],[0,0,1,1,0.8],[0,1,0,0,0.8],[0,1,0,1,0.8],[0,1,1,0,0.8],[0,1,1,1,0.8],[1,0,0,0,0.8],[1,0,0,1,0.8],[1,0,1,0,0.8],[1,0,1,1,0.8],[1,1,0,0,0.8],[1,1,0,1,0.8],[1,1,1,0,0.8],[1,1,1,1,0.8]])
d_output=np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])      #desired output
delw=np.array([[]])     #change in weight
h=np.arange(0,np.shape(d_output)[1])  #for tracking desired and actual output
x=np.arange(0,np.shape(d_output)[1]) #for interating on input array
d_output[d_output<0]=0
bias=0.8
count=0
op=np.array([[]])

w=np.array([[0.5,-0.3,0.9,-0.5,-1]])  #weights
w=w.reshape(np.shape(inputsys)[1],1)

def netcal():
    y=np.dot(inputsys,w)
    y=np.around(y,1)
    y=y.T
    return y
def delwcal(x):
    delw=np.dot(np.dot(0.1,(d_output[0,x]-b[0,x])),inputsys[x,:-1])  #delw=c*r*x leaving the bias part from input
    delw=delw.reshape(np.shape(delw)[0],1)                           #c->learning rate=0.1,r=(desired-output),x=input
    delw=np.append(delw,[[0]],axis=0)
    return delw

y=netcal()
b=y>=0   #if greater than equal then True or else False
b=b.astype(int)  #converting to integer type
h=np.equal(b,d_output) #checking if actual matches desired
h=h.astype(int)
while 0 in h:
    count=count+1

    #print(h)
    for i in x:
        delw=delwcal(i)
        w=w+delw                            #w_new=w_old+c*r*x
        print("Weights:\n {}".format(w))
        y=netcal()
        b=y>=0
        b=b.astype(int)
        h=np.equal(b,d_output)
        h=h.astype(int)
        print(h)
    print("number of iteration {}".format(count))

while True:

    for i in range(np.shape(inputsys)[1]):
        if i==np.shape(inputsys)[1]-1:
            val=float(input("enter the bias(trained): "))   #taking weights user input
            op=np.append(op,val)
        else:
            val=float(input("inputs to net: "))           #taking bias user input
            op=np.append(op,val)
    op.reshape(1,np.shape(inputsys)[1])
    #print(op)
    lastop=np.ones((1,np.shape(inputsys)[1]))

    result=np.around(np.dot(op,w),1)
    check=np.equal(op,lastop)
    check=check.astype(int)
    if 0 in check:
         print(op)
         print(result)
    lastop=op
    if result>=0:     #RELU activation
         result=1
    else:
         result=0
    op=np.delete(op,range(np.shape(inputsys)[1]),0)
    print(result)
