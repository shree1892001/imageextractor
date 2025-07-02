import React from 'react';
import Page from './Page';

interface PdfRendererProps {
  pdfData: any;
}

const PdfRenderer: React.FC<PdfRendererProps> = ({ pdfData }) => {
  return (
    <div>
      {pdfData.pages.map((page) => (
        <Page key={page.pageNumber} pageNumber={page.pageNumber} elements={page.elements} />
      ))}
    </div>
  );
};

export default PdfRenderer;