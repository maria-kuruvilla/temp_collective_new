import numpy as np

def position(tr): #shape returns tr.s.shape
    return(tr.s)


def speed(tr): #speed(tr).shape returns tr.speed.shape - 2
    v = (position(tr)[2:] - position(tr)[:-2]) / 2
    b = np.linalg.norm(v, axis=-1)
    return(b*60)

def acceleration(tr): #shape returns tr.acceleration.shape - 2
    a = position(tr)[2:] - 2 * position(tr)[1:-1] + position(tr)[:-2]
    aa = np.linalg.norm(a, axis=-1)  
    return(aa*3600)
        

def filter_speed_low_pass(tr, roi1 = 30, roi2 = 3340):
    speed_mask = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), speed(tr),copy=False)
    
    return(speed_mask)         


def threshold0(tr, j, roi1 = 30, roi2 = 3340, t = 10):
    
    list2 = [i for i, value in enumerate(filter_speed_low_pass(tr, roi1, roi2)[:,j]) if value > t]
    list2.insert(0,100000000)
    list1 = [value for i,value in enumerate(list2[1:]) if  (value != (list2[i]+1))]
        
    return(list1)

def threshold1(tr,j,loom, n=0, roi1 = 30, roi2 = 3340, t = 10):
    list1 = threshold0(tr, j, roi1 , roi2 , t)
    
    list2 = [value for i, value in enumerate(list1[:]) if value < (loom[n] + 700) and value > (loom[n]+500) ]
    
    return(list2)

def threshold2(tr, j, roi1 = 30, roi2 = 3340, t = 5):
    
    
    list2 = [i for i, value in enumerate(filter_speed_low_pass(tr, roi1, roi2)[:,j]) if value < t]
        
        
    return(list2)


def startle_size(tr, loom, n, roi1 = 30, roi2 = 3340, t1 = 10, t2 = 5):

    distance = np.empty([tr.number_of_individuals, 1])
    distance.fill(np.nan)

    perc99 = np.empty([tr.number_of_individuals, 1])
    perc99.fill(np.nan)

    perc90 = np.empty([tr.number_of_individuals, 1])
    perc90.fill(np.nan)

    perc50 = np.empty([tr.number_of_individuals, 1])
    perc50.fill(np.nan)

    avg = np.empty([tr.number_of_individuals, 1])
    avg.fill(np.nan)



    for ind in range(tr.number_of_individuals):
        speed_data = np.empty([1,])  
        speed_data.fill(np.nan)
        a = threshold1(tr,ind, loom,n)
        b = threshold2(tr,ind)

        c = []
        for i in a:
            for j in b:
                if j>i:
                    c.append(j)
                    break

        else:
            distance_mult = np.empty([len(a), 1])
            distance_mult.fill(np.nan)
            for k in range(len(a)):
                speed_data = np.r_[speed_data,filter_speed_low_pass(tr, roi1, roi2)[a[k]:c[k],ind].compressed()]
                #distance_mult[k] = np.sum(speed_data)
            #print(speed_data)
            distance[ind] = np.nansum(speed_data)
            perc99[ind] = np.nanpercentile(speed_data,99)
            perc90[ind] = np.nanpercentile(speed_data,90)
            perc50[ind] = np.nanpercentile(speed_data,50)
            avg[ind] = np.nanmean(speed_data)
    print(distance)
    return(
        np.nanmean(distance), np.nanmean(perc99), np.nanmean(perc90), 
        np.nanmean(perc50), np.nanmean(avg))
