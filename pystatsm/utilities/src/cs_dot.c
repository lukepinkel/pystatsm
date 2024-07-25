void cs_dot(const double *ax, const int *ai, const int anz, 
            const double *bx, const int *bi, const int bnz, 
            double *cx){
    int i=0, j=0;
    while (i<anz && j < bnz){
        
        if(ai[i] < bi[j]){
            i++;
        }
        else if (ai[i] > bi[j]){
            j++;
        }
        else{
            *cx += ax[i] * bx[j];
            i++;
            j++;
        }
    
    }
            
}