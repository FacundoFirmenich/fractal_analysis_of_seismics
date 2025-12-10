# Streamlit Deployment Guide - Pan-American Seismic Fractal Analysis
**Date**: 2025-12-09  
**Status**: READY FOR DEPLOYMENT  

---

## 🚀 STREAMLIT CLOUD DEPLOYMENT (RECOMMENDED)

### Prerequisites
- GitHub repository with all code
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### Steps

1. **Push code to GitHub** (if not already done):
```bash
git add .
git commit -m "Ready for Streamlit deployment"
git push origin main
```

2. **Create `requirements.txt`** (already exists in repo):
```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
requests>=2.26.0
joblib>=1.1.0
leidenalg>=0.8.0
igraph>=0.9.0
streamlit>=1.28.0
```

3. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Click "New app"
   - Repository: `[YOUR_USERNAME]/PanAmericanPaper`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

4. **App will be live at**:
   ```
   https://[your-app-name].streamlit.app
   ```

---

## 🖥️ LOCAL TESTING (Before deployment)

```bash
# Install Streamlit
pip install streamlit

# Run locally
streamlit run streamlit_app.py

# App will open at http://localhost:8501
```

---

## ⚙️ CONFIGURATION OPTIONS

### Current Features (streamlit_app.py)
- ✅ Region presets (San Andreas, Cascadia, Cocos) + Custom
- ✅ Date range selector (default: last 15 years)
- ✅ Magnitude threshold slider (2.0-5.0, default 2.4)
- ✅ Bootstrap iterations selector (50, 100, **200** - default)
- ✅ Real-time USGS data fetch
- ✅ 4 tabs: Correlation Integral, Rényi Spectrum, TGS, Results Table
- ✅ Interactive visualizations
- ✅ CSV download

### Performance
- **Fast regions** (N < 5,000 events): ~30-60 seconds
- **Large regions** (N ~ 20,000 events): ~2-5 minutes
- Bootstrap 200 iterations: publication-quality uncertainty

---

## 📦 DEPLOYMENT CHECKLIST

- [x] `streamlit_app.py` complete and tested
- [x] `requirements.txt` includes all dependencies
- [x] `sfa/` package importable
- [x] Default bootstrap=200 for publication quality
- [x] Error handling for USGS API failures
- [x] Progress indicators for long computations
- [x] Download functionality for results
- [ ] GitHub repository public (or private with access)
- [ ] Streamlit Cloud account created
- [ ] Deploy button clicked

---

## 🔧 TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'sfa'"
**Solution**: Ensure `sfa/` directory is in repository root and contains `__init__.py`

### Issue: "USGS API timeout"
**Solution**: App includes retry logic. User can adjust date range to reduce event count.

### Issue: "Streamlit Cloud resource limits"
**Solution**: For very large regions (N > 50,000), recommend local execution or reduce bootstrap to 100.

### Issue: "Slow computation"
**Solution**: Display progress bar (already implemented). Users see real-time status.

---

## 🌐 PUBLIC URL EXAMPLE

After deployment, share:
```
https://pan-american-fractal.streamlit.app
```

Users can:
1. Select region
2. Configure parameters
3. Click "Compute D₂"
4. View interactive results
5. Download CSV

---

## 📊 MONITORING

Streamlit Cloud provides:
- Real-time logs
- Usage analytics
- Performance metrics

Access at https://share.streamlit.io/[your-app]/analytics

---

## ✅ READY STATUS

**App Status**: ✅ PRODUCTION READY  
**Bootstrap**: ✅ 200 iterations default  
**UI**: ✅ Professional 4-tab layout  
**Data**: ✅ Real-time USGS integration  
**Download**: ✅ CSV export functional  

**Next Action**: Push to GitHub → Deploy on Streamlit Cloud → Share public URL
