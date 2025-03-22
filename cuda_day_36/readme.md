I am reading the paper "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" and trying to implement it in CUDA, I keep failing as of now. 
To understand the schematic and how would i be able to code that is still unclear to me. So I have uploaded this version in pytorch and got output. 

### Below is the output of my training for 20 epochs:

Unique characters: 65
Starting training...
Epoch 1, Iteration 20/100, Loss: 3.5550
Epoch 1, Iteration 40/100, Loss: 3.4252
Epoch 1, Iteration 60/100, Loss: 3.3380
Epoch 1, Iteration 80/100, Loss: 3.1209
Epoch 1, Iteration 100/100, Loss: 2.9780
Epoch 1 completed. Average Loss: 3.3664

--- Generated Text at Epoch 1 ---
To be, or not to be: that is the question:
Wue myodTeCwaIlre?lecbL SeI!e  c?nepMheJag cHfne

oachthJh u;hahr fOas ttxanBiJwhera thaWwJ

anNn.wu
Epoch 2, Iteration 20/100, Loss: 3.0762
Epoch 2, Iteration 40/100, Loss: 2.9856
Epoch 2, Iteration 60/100, Loss: 2.8354
Epoch 2, Iteration 80/100, Loss: 2.8811
Epoch 2, Iteration 100/100, Loss: 2.8294
Epoch 2 completed. Average Loss: 2.9545

--- Generated Text at Epoch 2 ---
To be, or not to be: that is the question:

hk 
-nem.
Mgve h Ihen,eth

embehMeelJ't, snd m  agqTthl tLon,  wkollOUTx m  wer,?AcmeN:
RIY; &poiul
Epoch 3, Iteration 20/100, Loss: 2.8360
Epoch 3, Iteration 40/100, Loss: 2.8029
Epoch 3, Iteration 60/100, Loss: 2.6008
Epoch 3, Iteration 80/100, Loss: 2.7286
Epoch 3, Iteration 100/100, Loss: 2.8701
Epoch 3 completed. Average Loss: 2.7657

--- Generated Text at Epoch 3 ---
To be, or not to be: that is the question:
nk.amep tits 'Be te d tsto prqbe wotit ba
Der whitenei
k!
Gsinyucp tosFacolI tierin?leomatRe siCbl t
Epoch 4, Iteration 20/100, Loss: 2.6966
Epoch 4, Iteration 40/100, Loss: 2.6694
Epoch 4, Iteration 60/100, Loss: 2.7340
Epoch 4, Iteration 80/100, Loss: 2.5285
Epoch 4, Iteration 100/100, Loss: 2.5742
Epoch 4 completed. Average Loss: 2.6883

--- Generated Text at Epoch 4 ---
To be, or not to be: that is the question:
Whenkr?
ThRt nd gho
zif:

SM: s omend, mo, d  Il e sat'ro
Hant st s g fjowld aJasE:
ald thowecoweedu
Epoch 5, Iteration 20/100, Loss: 2.5959
Epoch 5, Iteration 40/100, Loss: 2.6292
Epoch 5, Iteration 60/100, Loss: 2.5555
Epoch 5, Iteration 80/100, Loss: 2.5785
Epoch 5, Iteration 100/100, Loss: 2.6514
Epoch 5 completed. Average Loss: 2.6158

--- Generated Text at Epoch 5 ---
To be, or not to be: that is the question:
Yopbond.
Drave be Homd.
Thace m'refedl t, wJu, d t tole f be,
LgU3uillou t ty cbyeCooe wit iXaUMoucl
Epoch 6, Iteration 20/100, Loss: 2.6597
Epoch 6, Iteration 40/100, Loss: 2.5505
Epoch 6, Iteration 60/100, Loss: 2.5048
Epoch 6, Iteration 80/100, Loss: 2.5132
Epoch 6, Iteration 100/100, Loss: 2.5716
Epoch 6 completed. Average Loss: 2.5832

--- Generated Text at Epoch 6 ---
To be, or not to be: that is the question:
Rr gf buttU&

itil it li tou uThot d 


pt l IsveroV:
LNI t id 


-mentE:
:y r te n s okan infasth t
Epoch 7, Iteration 20/100, Loss: 2.6196
Epoch 7, Iteration 40/100, Loss: 2.5096
Epoch 7, Iteration 60/100, Loss: 2.4792
Epoch 7, Iteration 80/100, Loss: 2.7961
Epoch 7, Iteration 100/100, Loss: 2.5108
Epoch 7 completed. Average Loss: 2.5433

--- Generated Text at Epoch 7 ---
To be, or not to be: that is the question:
urd Eo s soPougl fes t g an s ve the d
Fenglcollorar pano morrgh sodathe, :bon; pauy Ioufly ; G i in
Epoch 8, Iteration 20/100, Loss: 2.5322
Epoch 8, Iteration 40/100, Loss: 2.5997
Epoch 8, Iteration 60/100, Loss: 2.5923
Epoch 8, Iteration 80/100, Loss: 2.4764
Epoch 8, Iteration 100/100, Loss: 2.5118
Epoch 8 completed. Average Loss: 2.5083

--- Generated Text at Epoch 8 ---
To be, or not to be: that is the question:
Tg s t t knm me tt I ghZy; thyoofaspcte that ias l? tsthdespot;
ch t be t nve td wo thdayhRhess so l
Epoch 9, Iteration 20/100, Loss: 2.5578
Epoch 9, Iteration 40/100, Loss: 2.5142
Epoch 9, Iteration 60/100, Loss: 2.4276
Epoch 9, Iteration 80/100, Loss: 2.4973
Epoch 9, Iteration 100/100, Loss: 2.5520
Epoch 9 completed. Average Loss: 2.4999

--- Generated Text at Epoch 9 ---
To be, or not to be: that is the question:
Ou at ou fo t
Mo,
Nopfo I
an ?
Hy, moac be;
HQ wnt tu feans, acath as IUTYoo t itouv;

BE-tFog f je 
Epoch 10, Iteration 20/100, Loss: 2.3412
Epoch 10, Iteration 40/100, Loss: 2.6428
Epoch 10, Iteration 60/100, Loss: 2.5617
Epoch 10, Iteration 80/100, Loss: 2.4790
Epoch 10, Iteration 100/100, Loss: 2.4261
Epoch 10 completed. Average Loss: 2.4713

--- Generated Text at Epoch 10 ---
To be, or not to be: that is the question:
B. rothiathourn bipe thaven blee wouthiriaphin
ave hanobe  thos,
Na.
Bon hitolon.e sthid la hep ho  
Epoch 11, Iteration 20/100, Loss: 2.5856
Epoch 11, Iteration 40/100, Loss: 2.3730
Epoch 11, Iteration 60/100, Loss: 2.4875
Epoch 11, Iteration 80/100, Loss: 2.5624
Epoch 11, Iteration 100/100, Loss: 2.4257
Epoch 11 completed. Average Loss: 2.4539

--- Generated Text at Epoch 11 ---
To be, or not to be: that is the question:
Tho booo thover fo igo irivinglaco ge mar wnormorow'd! tharot md th ofotoor thor!
Sod I opo,
Thetof 
Epoch 12, Iteration 20/100, Loss: 2.4050
Epoch 12, Iteration 40/100, Loss: 2.4220
Epoch 12, Iteration 60/100, Loss: 2.6843
Epoch 12, Iteration 80/100, Loss: 2.3614
Epoch 12, Iteration 100/100, Loss: 2.4559
Epoch 12 completed. Average Loss: 2.4391

--- Generated Text at Epoch 12 ---
To be, or not to be: that is the question:
Son an h' the eShiandgorchee l! bu aacow, ang desthasousthothamenDofo ber ee ige,
Round shard ans yc
Epoch 13, Iteration 20/100, Loss: 2.5667
Epoch 13, Iteration 40/100, Loss: 2.3406
Epoch 13, Iteration 60/100, Loss: 2.4547
Epoch 13, Iteration 80/100, Loss: 2.3740
Epoch 13, Iteration 100/100, Loss: 2.5012
Epoch 13 completed. Average Loss: 2.4360

--- Generated Text at Epoch 13 ---
To be, or not to be: that is the question:
Tictein sedekesth tau tit ke t t to t mereato ile st ste win sot t, n t he to my lod ngis t wiu t to
Epoch 14, Iteration 20/100, Loss: 2.3714
Epoch 14, Iteration 40/100, Loss: 2.3952
Epoch 14, Iteration 60/100, Loss: 2.6421
Epoch 14, Iteration 80/100, Loss: 2.3675
Epoch 14, Iteration 100/100, Loss: 2.4079
Epoch 14 completed. Average Loss: 2.4079

--- Generated Text at Epoch 14 ---
To be, or not to be: that is the question:
dethamisatod,
I f pond; nothomo cost sthe hind;
He ig gadave
Anon m.
Youcou?
Tosh so fnd nhe hod tof
Epoch 15, Iteration 20/100, Loss: 2.3716
Epoch 15, Iteration 40/100, Loss: 2.2782
Epoch 15, Iteration 60/100, Loss: 2.2569
Epoch 15, Iteration 80/100, Loss: 2.3716
Epoch 15, Iteration 100/100, Loss: 2.3399
Epoch 15 completed. Average Loss: 2.3825

--- Generated Text at Epoch 15 ---
To be, or not to be: that is the question:
Te cho fo t m mee and e the heachack thThe tethin's'dmecean as s tinI ban,
Th tond shethey ngehen my
Epoch 16, Iteration 20/100, Loss: 2.3233
Epoch 16, Iteration 40/100, Loss: 2.3789
Epoch 16, Iteration 60/100, Loss: 2.4575
Epoch 16, Iteration 80/100, Loss: 2.4679
Epoch 16, Iteration 100/100, Loss: 2.4380
Epoch 16 completed. Average Loss: 2.3922

--- Generated Text at Epoch 16 ---
To be, or not to be: that is the question:
Thid d!, bur'd d theamhois asasary'e:
ood mydotue ofo d.
Thas y se! ssyodGolde dld t d s: ldothe'd, 
Epoch 17, Iteration 20/100, Loss: 2.2911
Epoch 17, Iteration 40/100, Loss: 2.2847
Epoch 17, Iteration 60/100, Loss: 2.3754
Epoch 17, Iteration 80/100, Loss: 2.4067
Epoch 17, Iteration 100/100, Loss: 2.4073
Epoch 17 completed. Average Loss: 2.3857

--- Generated Text at Epoch 17 ---
To be, or not to be: that is the question:
Natu t f be
Yoolica tldAf tthoume hate
Tateawig; mit b be ade d ghouovateootiligt t te in thyo I .
T
Epoch 18, Iteration 20/100, Loss: 2.4289
Epoch 18, Iteration 40/100, Loss: 2.3958
Epoch 18, Iteration 60/100, Loss: 2.4217
Epoch 18, Iteration 80/100, Loss: 2.1896
Epoch 18, Iteration 100/100, Loss: 2.4359
Epoch 18 completed. Average Loss: 2.3483

--- Generated Text at Epoch 18 ---
To be, or not to be: that is the question:
TI
Bupobe tonct.
To is iain he theat ghthe owdes d othinthtas s f; m. th an d tothe cout wouse-atot 
Epoch 19, Iteration 20/100, Loss: 2.2134
Epoch 19, Iteration 40/100, Loss: 2.3794
Epoch 19, Iteration 60/100, Loss: 2.2583
Epoch 19, Iteration 80/100, Loss: 2.3395
Epoch 19, Iteration 100/100, Loss: 2.3747
Epoch 19 completed. Average Loss: 2.3695

--- Generated Text at Epoch 19 ---
To be, or not to be: that is the question:
ghe t t tho cho t agh yo thy beion he dit ct bh
Th thenno, Bu y t gho thit, thoake;
Thteaad buore it
Epoch 20, Iteration 20/100, Loss: 2.3517
Epoch 20, Iteration 40/100, Loss: 2.4588
Epoch 20, Iteration 60/100, Loss: 2.3993
Epoch 20, Iteration 80/100, Loss: 2.4826
Epoch 20, Iteration 100/100, Loss: 2.3248
Epoch 20 completed. Average Loss: 2.3686

--- Generated Text at Epoch 20 ---
To be, or not to be: that is the question:
To toKit.
Thesobeok m!d owo boke-poi! tee s: te u wooke.
Towoco jon at hitotante Ron lot de the the 
