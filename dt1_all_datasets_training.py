import argparse
import sys, os

import numpy as np
from sklearn.externals import joblib
import scipy


if __name__ == '__main__':
    
    try:
        parser = argparse.ArgumentParser(description='baseline for predicting labels')

        parser.add_argument('-d',
                            default='.',
                            help='Directory with datasets in SVMLight format')

        parser.add_argument('-id', type=int,
                            choices=[1,2,3],
                            help='Dataset id')

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

        n_features = 100 ## dataset id = 3
        if args.id == 1:
            n_features = 200
        elif args.id == 2:
            n_features = 200
        
        fname_trn1 = os.path.join(args.d, "dt1_trn_lsa.svm") 
        fname_trn2 = os.path.join(args.d, "dt2.trn.svm") 
        fname_trn3 = os.path.join(args.d, "dt3_trn_lsa.svm") 
        #fname_trn4 = os.path.join(args.d, "dt2.vld.svm") 
        #fname_trn5 = os.path.join(args.d, "dt3_vld_lsa.svm") 




        fname_vld = os.path.join(args.d, "dt1_vld_lsa.svm")
        fname_tst = os.path.join(args.d, "dt1_tst_lsa.svm")

        fname_vld_lbl = os.path.join(args.d, "dt%d.%s.lsa.LinearSVC.lbl" % (args.id, "vld"))
        fname_tst_lbl = os.path.join(args.d, "dt%d.%s.lsa.LinearSVC.lbl" % (args.id, "tst"))

        fname_vld_pred = os.path.join(args.d, "dt%d.%s.lsa.LinearSVC.pred" % (args.id, "vld"))
        fname_tst_pred = os.path.join(args.d, "dt%d.%s.lsa.LinearSVC.pred" % (args.id, "tst"))
        
       
        
        ### reading labels
        from sklearn.datasets import dump_svmlight_file, load_svmlight_file
        data_trn1, lbl_trn1 = load_svmlight_file(fname_trn1, n_features=200, zero_based=True)
        data_trn2, lbl_trn2 = load_svmlight_file(fname_trn2, n_features=200, zero_based=True)
        data_trn3, lbl_trn3 = load_svmlight_file(fname_trn3, n_features=200, zero_based=True)
        #data_trn4, lbl_trn4 = load_svmlight_file(fname_trn4, n_features=200, zero_based=True)
        #data_trn5, lbl_trn5 = load_svmlight_file(fname_trn5, n_features=200, zero_based=True)


        
        data_trn = scipy.sparse.vstack( (data_trn1, data_trn2, data_trn3) )

        lbl_trn = np.hstack((lbl_trn1, lbl_trn2, lbl_trn3))
        

        #data_trn = []
        #data_trn = data_trn1
        #for i in range(1,10900):
        #    data_trn[i-1,1:] = data_trn2[i-1,1:]

        #for j in range(1,8200):
         #   data_trn[21801:30000,1:] = data_trn3


        

        #data_trn4 = np.concatenate(data_trn1, data_trn2)
        #data_trn = np.concatenate(data_trn4, data_trn3)

        #lbl_trn4 = np.concatenate(lbl_trn1, lbl_trn2)
        #lbl_trn = np.concatenate(lbl_trn4, lbl_trn3)

        data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=n_features, zero_based=True)
        data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=n_features, zero_based=True)

        print data_trn.shape
        print lbl_trn.shape
        print data_vld.shape
        print lbl_vld.shape
        print data_tst.shape
        print lbl_tst.shape
        
        ### perform grid search using validation samples
        from sklearn.grid_search import ParameterGrid
        from sklearn.svm import LinearSVC, SVC, SVR
        from sklearn.metrics import mean_squared_error, accuracy_score
        dt1_grid = [{'kernel': ['rbf'], 'C': [1, 10, 850],
                     'gamma': [0.1, 1.0, 60]}]

        dt2_grid = [{'kernel': ['rbf'], 'C': [1.0, 100.0],
                     'gamma': [0.1, 1.0]}]

        dt3_grid = [{'kernel': ['rbf'], 'C': [1.0, 100.0, 10000.0],
                     'gamma': [0.1, 1.0, 10.0]}]

        grids = (None, dt1_grid, dt2_grid, dt3_grid)
        classifiers = (None, SVC, SVC, SVR)
        metrics = (None, accuracy_score, accuracy_score, mean_squared_error)
        str_formats = (None, "%d", "%d", "%.6f")
        #LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0,

        grid_obj=grids[args.id]
        cls_obj=classifiers[args.id]
        metric_obj=metrics[args.id]
        
        best_param = None
        best_score = None
        best_svc = None
        
        for one_param in ParameterGrid(grid_obj):
            cls = cls_obj(**one_param)
            cls.fit(data_trn, lbl_trn)
            
            one_score = metric_obj(lbl_vld, cls.predict(data_vld))
            
            print ("param=%s, score=%.6f" % (repr(one_param),one_score))
            
            if ( best_score is None or 
                 (args.id < 3 and best_score < one_score) or
                 (args.id == 3 and best_score > one_score) ):
                best_param = one_param
                best_score = one_score
                best_svc = cls
            
        pred_vld = best_svc.predict(data_vld)
        pred_tst = best_svc.predict(data_tst)
        
        print ("Best score for vld: %.6f" % (metric_obj(lbl_vld, pred_vld),))
        print ("Best score for tst: %.6f" % (metric_obj(lbl_tst, pred_tst),))
        
        np.savetxt(fname_vld_pred, pred_vld, delimiter='\n', fmt=str_formats[args.id])
        np.savetxt(fname_tst_pred, pred_tst, delimiter='\n', fmt=str_formats[args.id])
        
        np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=str_formats[args.id])
        np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=str_formats[args.id])

    except Exception, exc:
        import traceback
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))






