"""Localized economic, social and logistics charts."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

LABELS = {
    'fr': ['Analyse économique et sociale','Coût conventionnel estimé vs EcoRoute','Économies par douar','Isolement et économies','Remplissage des camions','Conventionnel estimé','EcoRoute (collecte locale incluse)','Coût (MAD)','Douar','Économies (%)','Score relatif d’isolement (%)','Camion','Chargement (%)','Objectif minimal','Les économies reposent sur des coûts conventionnels estimés, non sur des tarifs vérifiés sur le terrain.','Production (kg)','Redistribution (MAD)','La tarification est irréalisable : seuls les indicateurs logistiques sont affichés.'],
    'en': ['Economic and social analysis','Estimated conventional cost vs EcoRoute','Savings by settlement','Isolation and savings','Truck utilization','Estimated conventional','EcoRoute (including local collection)','Cost (MAD)','Settlement','Savings (%)','Relative isolation score (%)','Truck','Utilization (%)','Minimum target','Savings rely on estimated conventional costs, not field-verified prices.','Production (kg)','Redistribution (MAD)','Pricing is infeasible: only logistics indicators are displayed.'],
    'ar': ['التحليل الاقتصادي والاجتماعي','التكلفة التقليدية المقدرة مقارنة مع EcoRoute','التوفير حسب الدوار','العزلة والتوفير','نسبة ملء الشاحنات','التكلفة التقليدية المقدرة','EcoRoute (شاملة التجميع المحلي)','التكلفة (درهم)','الدوار','التوفير (%)','مؤشر العزلة النسبي (%)','الشاحنة','نسبة الملء (%)','الهدف الأدنى','تعتمد نسب التوفير على تكاليف تقليدية تقديرية وليست أسعارا مؤكدة ميدانيا.','الإنتاج (كغ)','إعادة التوزيع (درهم)','التسعير غير قابل للتحقيق: تعرض المؤشرات اللوجستية فقط.']
}


def build_charts(result, language='fr', minimum_target=None):
    l = LABELS[language]
    figures = {}
    if result['pricing']['feasible']:
        p = result['pricing']['results'].copy()
        p['isolation_pct'] = p['isolation_score'] * 100
        p['settlement_label'] = p['name'].astype(str)
        duplicate = p['settlement_label'].duplicated(keep=False)
        p.loc[duplicate, 'settlement_label'] += ' — ' + p.loc[duplicate, 'osm_id'].astype(str)
        k = result['kpis']
        fig = go.Figure(go.Bar(x=[l[5], l[6]], y=[k['total_baseline_cost'], k['total_effective_cost']], marker_color=['#64748B','#16803D'], texttemplate='%{y:,.2f}', textposition='outside'))
        fig.update_layout(title=l[1], yaxis_title=l[7], xaxis_title=None)
        fig.update_yaxes(rangemode='tozero')
        figures['cost'] = fig
        p = p.sort_values('savings_pct')
        fig = px.bar(p, x='savings_pct', y='settlement_label', orientation='h', title=l[2], labels={'savings_pct':l[9], 'settlement_label':l[8]}, color_discrete_sequence=['#16803D'])
        fig.update_layout(height=max(350, min(1000, 100+24*len(p))), yaxis={'automargin':True})
        if minimum_target is not None:
            fig.add_vline(x=minimum_target, line_dash='dash', line_color='#B45309', annotation_text=l[13])
        figures['savings'] = fig
        fig = px.scatter(p, x='isolation_pct', y='savings_pct', size='quantity_kg', hover_name='settlement_label', hover_data={'solidarity_discount':':.2f','quantity_kg':True}, labels={'isolation_pct':l[10],'savings_pct':l[9],'quantity_kg':l[15],'solidarity_discount':l[16]}, title=l[3], color_discrete_sequence=['#16803D'], size_max=35)
        if minimum_target is not None:
            fig.add_hline(y=minimum_target, line_dash='dash', line_color='#B45309')
        figures['social'] = fig
    fleet = pd.DataFrame(result['fleet'])
    used = {int(r['truck_id']):r['load_kg'] for r in result['routing']['routes']}
    fleet['label'] = fleet['id'].map(lambda x: f'{l[11]} {x}')
    fleet['utilization'] = fleet.apply(lambda r: 100*used.get(int(r['id']),0)/r['capacity_kg'],axis=1)
    fig = px.bar(fleet, x='label', y='utilization', title=l[4], labels={'label':l[11],'utilization':l[12]}, color_discrete_sequence=['#2563EB'], text='utilization')
    fig.update_traces(texttemplate='%{y:.1f}%')
    fig.update_yaxes(range=[0,110])
    figures['fleet'] = fig
    for fig in figures.values():
        fig.update_layout(template='plotly_white', margin=dict(l=20,r=25,t=65,b=40), font=dict(family='Arial, sans-serif'), showlegend=False)
    return figures


def render_charts(result, language='fr', minimum_target=None):
    l = LABELS[language]
    st.divider()
    st.subheader(l[0])
    figures = build_charts(result, language, minimum_target)
    if 'cost' in figures:
        st.caption(l[14])
        a,b = st.columns(2)
        a.plotly_chart(figures['cost'], use_container_width=True, key='chart_cost')
        b.plotly_chart(figures['fleet'], use_container_width=True, key='chart_fleet')
        st.plotly_chart(figures['savings'], use_container_width=True, key='chart_savings')
        st.plotly_chart(figures['social'], use_container_width=True, key='chart_social')
    else:
        st.info(l[17])
        st.plotly_chart(figures['fleet'], use_container_width=True, key='chart_fleet')
