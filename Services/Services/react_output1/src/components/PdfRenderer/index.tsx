import React from 'react';
import Page from './Page';
import { pdfData } from './pdfData'; //Import your PDF data here

const PdfRenderer: React.FC = () => {
  return (
    <div>
      {pdfData.pages.map((page) => (
        <Page key={page.pageNumber} {...page} />
      ))}
    </div>
  );
};

export default PdfRenderer;