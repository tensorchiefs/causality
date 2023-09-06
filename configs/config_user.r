config_user = function(name){
  if(name=="beate_server"){
    DROPBOX <<- '/home/sick/causality_db/'  # overwrite globel var within fct
  }
  else if(name=="beate_windows"){
    DROPBOX <<- 'C:/Users/sick/dl Dropbox/beate sick/IDP_Projekte/DL_Projekte/shared_Oliver_Beate/Causality_2022/tram_DAG/'
  }
  else if(name=="oliver"){
    DROPBOX <<- '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/'
  }
  else{
    print("no valid user name")
    stop("no valid user name")
  }
}