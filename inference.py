from testbestmodel import Test

def main(typeModel,typeGRU):
    pathRootModel = "/home/ns14/Desktop/us-models"
    pathTestset   = "/home/ns14/Desktop/testset"
    pathSave      = "/home/ns14/Desktop/saveTest"
    frameSecond   = 25
    Test(pathRootModel,pathTestset,pathSave,\
    	 typeModel,typeGRU,frameSecond)

if __name__ == "__main__":
    #AlexNet
    main("alexnet","copyframe")
    main("alexnet","unidir")
    main("alexnet","bidir")
    #Densenet121
    main("densenet","copyframe")
    main("densenet","unidir")
    main("densenet","bidir")
    #InceptionV4
    main("inception","copyframe")
    main("inception","unidir")
    main("inception","bidir")
    #ResNet18
    main("resnet","copyframe")
    main("resnet","unidir")
    main("resnet","bidir")
    #Vgg-E
    main("vgg","copyframe")
    main("vgg","unidir")
    main("vgg","bidir")
    print("\n DONE!")
