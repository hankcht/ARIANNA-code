All data:               Circled data:         Above BL curve data:                      Max Partition:
stn 14:     963361                      52                            318               2
stn 17:     1542768                     62                           2326               3
stn 19:     608632                      38                            317               2
stn 30:     857496                      63                           5618               2
            3972257   total                                          8579   total

stn 13:     363336                      Nan                          2115               1
stn 15:     663245                      Nan                          1282               2
stn 18:     872176                      Nan                           957               2
            1898757   total                                          4354   total

CONFIRMED BL 
200S                                                         100S
Event2016_Stn14_1454540191.0_Chi0.66_SNR5.57.npy             Event2016_Stn15_1450734371.0_Chi0.57_SNR7.10.npy**
Event2016_Stn14_1455263868.0_Chi0.58_SNR7.68.npy            
                                                             Event2016_Stn18_1449861609.0_Chi0.80_SNR16.85.npy
Event2016_Stn17_1449861609.0_Chi0.68_SNR20.33.npy*           Event2016_Stn18_1450268467.0_Chi0.77_SNR13.90.npy    
Event2016_Stn17_1450734371.0_Chi0.65_SNR7.70.npy             Event2016_Stn18_1450734371.0_Chi0.71_SNR7.54.npy**
Event2016_Stn17_1457453131.0_Chi0.63_SNR6.57.npy             Event2016_Stn18_1455205950.0_Chi0.70_SNR7.32.npy
                                                             Event2016_Stn18_1455513662.0_Chi0.73_SNR9.26.npy
Event2016_Stn30_1449861609.0_Chi0.69_SNR9.62.npy*            Event2016_Stn18_1458294171.0_Chi0.73_SNR6.69.npy
Event2016_Stn30_1455205950.0_Chi0.64_SNR8.17.npy            
Event2016_Stn30_1455513662.0_Chi0.74_SNR16.57.npy           
Event2016_Stn30_1458294171.0_Chi0.61_SNR5.40.npy            

Notes:
- We have two different RCR efficiency. A1: shows percentage of events higher than the network output cut 
                                        B1: shows the percentage of weighted RCR above the BL curve 


A1:
                  + sim BL              (1)                
Train: sim RCR <
                  + "BL data events"    (2)

                  + sim BL              (1)                
Run:   sim RCR <
                  + "BL data events"    (2)

example: 1.2 is sim_data, 2.2 is data_data 

The Run part in A1: we create a histogram of events of non-trained events (a portion of the simulated events)
 
A2:
total circled events := circled events on Chi-SNR graph, these are events that passed the Noise Cut
selected events := certain circled events that we picked (Chi > 0.5)
passed events := events that passes the network output cut (value imported above)  
selected passed events := selected events that passed the network output cut    

After performing a quick check A1, now we see how the model actually interprets the events
We will run the model on:
    1. All sim data
    2. confirmed BL events
    3. All station data

A3, A4:
change model to run with
plotsim has 'individual' or 'combined' modes
in main, change partition 
change chunk if needed



Chi 2016 (BL template)
Chi Threshold(above the cut)    60:              65:             70:
stn 14:                             2000            976             314
stn 17:                             30579           20275           4443
stn 19:                             14852           10456           6732
stn 30:                             75617           25397           7803

stn 13:                             42844           14065           1123
stn 15:                             69682           40100           19840   
stn 18:                             929             332             49

Different number of total events on SNR-Chi plot between old template and new template
(963361-480056)+(1542768-1095183)+(608632-263134)+(857496-591144)+(363336-202840)+(663245-265882)+(872176-515984) = 2456791



~1000 Above Curve events curve settings:
13: x1, y1 = 4.5, 0.5
    x2, y2 = 9, 0.66
    x3, y3 = 20, 0.8

14: x1, y1 = 4.5, 0.5
    x2, y2 = 23, 0.8

15: x1, y1 = 5, 0.5
    x2, y2 = 10, 0.685
    x3, y3 = 20, 0.78
    x4, y4 = 40, 0.84

17: x1, y1 = 4, 0.5
    x2, y2 = 4.5, 0.6
    x3, y3 = 10, 0.75
    x4, y4 = 23, 0.8

18: x1, y1 = 5, 0.6
    x2, y2 = 30, 0.8

19: x1, y1 = 4.7, 0.5
    x2, y2 = 10, 0.65
    x3, y3 = 16, 0.7
    x4, y4 = 20, 0.77

30: x1, y1 = 4.5, 0.53
    x2, y2 = 10, 0.72
    x3, y3 = 20, 0.8

















