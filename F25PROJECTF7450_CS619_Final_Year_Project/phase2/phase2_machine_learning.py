"""
  CS619 – PHASE 2: PREPROCESSING + MACHINE LEARNING
  
"""
import os, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import SVC
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      classification_report, confusion_matrix)
warnings.filterwarnings('ignore')

def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

NUMERIC_COLS = [
    'Age','Semester','Total_Classes','Classes_Attended','Attendance_Percentage',
    'Assignment_1','Assignment_2','Assignment_3','Average_Assignment',
    'Quiz_1','Quiz_2','Average_Quiz','Midterm_Exam','Final_Exam',
    'Lab_Work_1','Lab_Work_2','Average_Lab','Project_Score','CGPA','Previous_GPA',
    'Study_Hours_Per_Week','Library_Visits','Online_Course_Hours',
    'Class_Participation','Group_Work_Score','Total_Score',
]
CATEGORICAL_COLS = ['Gender','Course','Section','Sports_Activity','Club_Membership']
DROP_ID_COLS     = ['Student_ID','Registration_No']

MODELS = {
    'Decision Tree'      : DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
    'SVM'                : SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
}

def run_preprocessing(data_path='processed_student_data.csv'):
    section("STEP 1 – DATA PREPROCESSING")
    df = pd.read_csv(data_path); df.columns = df.columns.str.strip()
    print(f"  Dataset loaded       : {df.shape[0]} records, {df.shape[1]} features")
    df = df.drop(columns=[c for c in DROP_ID_COLS if c in df.columns])

    le_dict = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df['Performance'])
    X = df.drop(columns=['Performance'])
    print(f"  Target classes       : {list(label_enc.classes_)}")

    num_present = [c for c in NUMERIC_COLS if c in X.columns]
    scaler = StandardScaler()
    X[num_present] = scaler.fit_transform(X[num_present])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print(f"  Train / Test split   : {len(X_train)} / {len(X_test)} samples")

    X_train.to_csv('phase2_X_train.csv',index=False)
    X_test.to_csv ('phase2_X_test.csv', index=False)
    pd.Series(y_train,name='Performance').to_csv('phase2_y_train.csv',index=False)
    pd.Series(y_test, name='Performance').to_csv('phase2_y_test.csv', index=False)
    pd.DataFrame({'class_index':range(len(label_enc.classes_)),'class_label':label_enc.classes_}).to_csv('phase2_label_map.csv',index=False)

    joblib.dump(scaler,    'student_scaler_v1.pkl')
    joblib.dump(label_enc, 'student_encoder_v1.pkl')
    joblib.dump(le_dict,   'student_feature_encoders_v1.pkl')
    return X_train, X_test, y_train, y_test, label_enc

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train); y_pred = model.predict(X_test)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_scores = cross_val_score(model,X_train,y_train,cv=cv,scoring='accuracy')
    return {'name':name,'model':model,'y_pred':y_pred,
            'accuracy':accuracy_score(y_test,y_pred),
            'precision':precision_score(y_test,y_pred,average='weighted',zero_division=0),
            'recall':recall_score(y_test,y_pred,average='weighted',zero_division=0),
            'f1_score':f1_score(y_test,y_pred,average='weighted',zero_division=0),
            'cv_mean':cv_scores.mean(),'cv_std':cv_scores.std()}

def run_models(X_train,X_test,y_train,y_test,label_enc,out_dir='.'):
    section("STEP 2 – MACHINE LEARNING MODELS")
    results=[]
    for name,model in MODELS.items():
        print(f"  Training : {name} …")
        res=evaluate_model(name,model,X_train,X_test,y_train,y_test)
        results.append(res)

    section("RESULTS SUMMARY")
    hdr=f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'CV Mean':>10} {'CV Std':>8}"
    print(hdr); print('-'*len(hdr))
    for r in results:
        print(f"{r['name']:<22} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['cv_mean']:>10.4f} {r['cv_std']:>8.4f}")

    best=max(results,key=lambda r:r['f1_score'])
    section(f"BEST MODEL → {best['name']}")
    print(f"  Accuracy : {best['accuracy']:.4f}  |  Precision : {best['precision']:.4f}  |  Recall : {best['recall']:.4f}  |  F1 : {best['f1_score']:.4f}  |  CV : {best['cv_mean']:.4f} ±{best['cv_std']:.4f}")

    joblib.dump(best['model'],'student_model_v1.pkl')

    section("GENERATING PLOTS")
    colors_cm=['Blues','Greens','Oranges','Reds']
    for r,cmap in zip(results,colors_cm):
        cm=confusion_matrix(y_test,r['y_pred'])
        fig,ax=plt.subplots(figsize=(8,6))
        sns.heatmap(cm,annot=True,fmt='d',cmap=cmap,xticklabels=label_enc.classes_,yticklabels=label_enc.classes_,ax=ax)
        ax.set_title(f"Confusion Matrix – {r['name']}",fontsize=13,fontweight='bold')
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted'); plt.tight_layout()
        fname=os.path.join(out_dir,f"cm_{r['name'].replace(' ','_').lower()}.png")
        plt.savefig(fname,dpi=150); plt.close()

    metrics=['accuracy','precision','recall','f1_score']
    labels=[r['name'] for r in results]; x=np.arange(len(labels)); width=0.2
    fig,ax=plt.subplots(figsize=(12,6))
    for i,(metric,color) in enumerate(zip(metrics,['#2196F3','#4CAF50','#FF9800','#E91E63'])):
        vals=[r[metric] for r in results]
        bars=ax.bar(x+i*width,vals,width,label=metric.capitalize(),color=color,alpha=0.85)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{val:.3f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x+width*1.5); ax.set_xticklabels(labels,rotation=15,ha='right')
    ax.set_ylim(0,1.15); ax.set_ylabel('Score')
    ax.set_title('Model Comparison',fontsize=13,fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(axis='y',alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'model_comparison.png'),dpi=150); plt.close()
    print("  Plots saved successfully.")

    pd.DataFrame([{'Model':r['name'],'Accuracy':round(r['accuracy'],4),'Precision':round(r['precision'],4),
                   'Recall':round(r['recall'],4),'F1_Score':round(r['f1_score'],4),
                   'CV_Mean':round(r['cv_mean'],4),'CV_Std':round(r['cv_std'],4),'Best':r['name']==best['name']} for r in results]
                ).to_csv(os.path.join(out_dir,'phase2_model_results.csv'),index=False)
    return results, best

def main(data_path='processed_student_data.csv',out_dir='.'):
    print("\n"+"="*60+"\n   CS619 – PHASE 2  (Preprocessing + Machine Learning)\n"+"="*60)
    os.makedirs(out_dir,exist_ok=True)
    X_train,X_test,y_train,y_test,label_enc=run_preprocessing(data_path)
    results,best=run_models(X_train,X_test,y_train,y_test,label_enc,out_dir)
    section("ARTEFACTS")
    for p in ['student_model_v1.pkl','student_scaler_v1.pkl','student_encoder_v1.pkl','student_feature_encoders_v1.pkl']:
        print(f"  {'✓' if os.path.exists(p) else '✗ MISSING'}  {p}")
    print("\n"+"="*60+f"\n  Phase 2 complete. Best model: {best['name']}\n"+"="*60+"\n")

if __name__=='__main__':
    main(data_path='processed_student_data.csv',out_dir='.')